"""
Main CIAO explainer implementation using Lightning framework.
"""

from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

from ciao.components.obfuscation.base import BaseObfuscation
from ciao.typing import Classifier, Image, SegmentMask


class CIAOFeatureSelector:
    """
    Core feature selection algorithm for CIAO method.
    Implements Algorithm 1 from the paper with mutual information scoring.
    """

    def __init__(
        self,
        model: Classifier,
        target_class: Optional[int] = None,
        eta: float = 0.01,
        radius_range: List[int] = [1, 2],
        adaptive_eta: bool = True,
    ):
        """Initialize CIAO feature selector."""
        self.model = model
        self.target_class = target_class
        self.base_eta = eta
        self.eta = eta
        self.radius_range = radius_range
        self.adaptive_eta = adaptive_eta
        self.model.eval()

    def select_features(
        self,
        image: Image,
        segments: SegmentMask,
        adjacency_graph: nx.Graph,
        obfuscator: BaseObfuscation,
    ) -> List[Set[int]]:
        """Main feature selection algorithm."""
        # Create surrogate dataset
        surrogate_data = self._create_surrogate_dataset(
            image, segments, adjacency_graph, obfuscator
        )

        # Apply feature selection algorithm
        feature_groups = self._feature_selection_algorithm(
            segments, adjacency_graph, surrogate_data
        )

        return feature_groups

    def _create_surrogate_dataset(
        self,
        image: Image,
        segments: SegmentMask,
        adjacency_graph: nx.Graph,
        obfuscator: BaseObfuscation,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create surrogate dataset as described in the paper."""
        # Convert to numpy array for processing
        if isinstance(image, torch.Tensor):
            # Move to CPU and convert to numpy
            image_np = image.detach().cpu().numpy()
            if len(image_np.shape) == 4:  # Remove batch dimension if present
                image_np = image_np.squeeze(0)
            if image_np.shape[0] == 3:  # CHW to HWC
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image

        # Get original prediction (pass original image, not numpy)
        original_pred = self._get_model_prediction(image)
        if self.target_class is None:
            self.target_class = int(original_pred.argmax().item())

        # Get all segments
        segment_ids = np.unique(segments)
        segment_ids = segment_ids[segment_ids >= 0]

        # Create local groups
        local_groups = []
        for segment_id in segment_ids:
            for radius in self.radius_range:
                group = self._get_local_group(segment_id, radius, adjacency_graph)
                if len(group) > 0:
                    local_groups.append(group)

        # Generate surrogate dataset
        X = []
        y = []
        deltas = []

        for group in local_groups:
            # Create binary feature vector
            feature_vector = np.zeros(len(segment_ids))
            for seg_id in group:
                if seg_id in segment_ids:
                    idx = np.where(segment_ids == seg_id)[0][0]
                    feature_vector[idx] = 1

            # Obfuscate and get prediction
            obfuscated_image = obfuscator.obfuscate_segments(
                image_np, segments, list(group)
            )
            obfuscated_pred = self._get_model_prediction(obfuscated_image)

            # Calculate delta
            target_idx = int(self.target_class)
            orig_score = original_pred[target_idx].item()
            obfus_score = obfuscated_pred[target_idx].item()
            delta = abs(orig_score - obfus_score)
            deltas.append(delta)
            X.append(feature_vector)

        # Create binary targets
        delta_threshold = np.mean(deltas)
        for delta in deltas:
            y.append(1 if delta >= delta_threshold else 0)

        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)

        # Adapt eta if enabled
        if self.adaptive_eta:
            self.eta = self._compute_adaptive_eta(X_array, y_array, deltas)

        return X_array, y_array

    def _compute_adaptive_eta(
        self, X: np.ndarray, y: np.ndarray, deltas: List[float]
    ) -> float:
        """Compute adaptive eta threshold."""
        if len(deltas) == 0:
            return self.base_eta

        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas)
        positive_ratio = np.mean(y) if len(y) > 0 else 0.5

        # Adapt eta based on data characteristics
        if delta_mean > 0:
            delta_factor = min(delta_mean * 0.8, self.base_eta)
        else:
            delta_factor = self.base_eta * 0.3

        if delta_std > 0 and delta_mean > 0:
            variance_factor = max(0.2, 1.0 - (delta_std / delta_mean) * 0.5)
        else:
            variance_factor = 0.5

        balance_factor = min(positive_ratio, 1 - positive_ratio) * 3
        if balance_factor < 0.3:
            balance_factor = 0.3
        else:
            balance_factor = min(balance_factor, 1.0)

        adaptive_eta = delta_factor * variance_factor * balance_factor * 0.7
        adaptive_eta = max(0.005, min(adaptive_eta, self.base_eta))

        return float(adaptive_eta)

    def _get_local_group(
        self, center_segment: int, radius: int, adjacency_graph: nx.Graph
    ) -> Set[int]:
        """Get local group within radius."""
        if center_segment not in adjacency_graph:
            return set()

        group = set()
        visited = set()
        queue = [(center_segment, 0)]

        while queue:
            current_node, distance = queue.pop(0)
            if current_node in visited:
                continue

            visited.add(current_node)
            if distance <= radius:
                group.add(current_node)

                if distance < radius:
                    for neighbor in adjacency_graph.neighbors(current_node):
                        if neighbor not in visited:
                            queue.append((neighbor, distance + 1))

        return group

    def _feature_selection_algorithm(
        self,
        segments: SegmentMask,
        adjacency_graph: nx.Graph,
        surrogate_data: Tuple[np.ndarray, np.ndarray],
    ) -> List[Set[int]]:
        """Feature selection algorithm main loop."""
        X, y = surrogate_data
        segment_ids = np.unique(segments)
        segment_ids = segment_ids[segment_ids >= 0]

        feature_groups = []
        remaining_features = set(range(len(segment_ids)))

        while remaining_features:
            # Find seed feature
            seed_idx = self._find_seed_feature(X, y, remaining_features)
            if seed_idx is None:
                break

            current_group = {seed_idx}
            remaining_features.remove(seed_idx)

            # Grow group iteratively
            while True:
                neighbors = self._get_group_neighbors(
                    current_group, segment_ids, adjacency_graph, remaining_features
                )

                if not neighbors:
                    break

                best_neighbor, best_score = self._find_best_neighbor(
                    X, y, current_group, neighbors
                )

                if best_neighbor is None or best_score < self.eta:
                    break

                current_group.add(best_neighbor)
                remaining_features.remove(best_neighbor)

            # Convert indices back to segment IDs
            segment_group = {segment_ids[idx] for idx in current_group}
            feature_groups.append(segment_group)

        return feature_groups

    def _get_model_prediction(self, image: Image) -> torch.Tensor:
        """Get model prediction for image."""
        # Convert to tensor if numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:  # H, W, C
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        elif isinstance(image, torch.Tensor):
            image_tensor = image.clone()
            if len(image_tensor.shape) == 3:  # Could be C,H,W or H,W,C
                # Check if it's H,W,C (channels last) and convert to C,H,W
                if image_tensor.shape[2] == 3:  # H,W,C format
                    image_tensor = image_tensor.permute(2, 0, 1)
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            elif len(image_tensor.shape) == 4:  # Already has batch dimension
                # Check if it's B,H,W,C and convert to B,C,H,W
                if image_tensor.shape[3] == 3:  # B,H,W,C format
                    image_tensor = image_tensor.permute(0, 3, 1, 2)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Ensure image is in correct device
        try:
            # Try to get device from model parameters
            params = getattr(self.model, "parameters", None)
            if params is not None:
                device = next(params()).device
                image_tensor = image_tensor.to(device)
            else:
                # Try direct device attribute
                device = getattr(self.model, "device", None)
                if device is not None:
                    image_tensor = image_tensor.to(device)
        except (StopIteration, AttributeError, TypeError):
            # Model has no parameters or device info, use CPU
            pass

        with torch.no_grad():
            prediction = self.model(image_tensor)
            if len(prediction.shape) > 1:
                prediction = torch.softmax(prediction, dim=1)
            else:
                prediction = torch.sigmoid(prediction)

        return prediction.squeeze(0)

    def _find_seed_feature(
        self, X: np.ndarray, y: np.ndarray, candidates: set
    ) -> Optional[int]:
        """Find best seed feature using mutual information."""
        if len(candidates) == 0 or len(set(y)) < 2:
            return None

        best_score = -1
        best_feature = None

        for feature_idx in candidates:
            if len(set(X[:, feature_idx])) < 2:
                continue

            score = mutual_info_classif(
                X[:, [feature_idx]], y, discrete_features=True, random_state=42
            )[0]

            if score > best_score:
                best_score = score
                best_feature = feature_idx

        return best_feature

    def _get_group_neighbors(
        self,
        group: Set[int],
        segment_ids: np.ndarray,
        adjacency_graph: nx.Graph,
        remaining_features: Set[int],
    ) -> Set[int]:
        """Get neighbors of current group."""
        neighbors = set()

        for feature_idx in group:
            segment_id = segment_ids[feature_idx]
            if segment_id in adjacency_graph:
                for neighbor_segment in adjacency_graph.neighbors(segment_id):
                    neighbor_idx = np.where(segment_ids == neighbor_segment)[0]
                    if len(neighbor_idx) > 0 and neighbor_idx[0] in remaining_features:
                        neighbors.add(neighbor_idx[0])

        return neighbors

    def _find_best_neighbor(
        self, X: np.ndarray, y: np.ndarray, current_group: Set[int], neighbors: Set[int]
    ) -> Tuple[Optional[int], float]:
        """Find best neighbor to add to group."""
        if len(set(y)) < 2:
            return None, 0.0

        best_score = -1
        best_neighbor = None

        # Current group features
        group_features = np.sum(X[:, list(current_group)], axis=1) > 0

        for neighbor_idx in neighbors:
            # Combined features (group OR neighbor)
            neighbor_feature = X[:, neighbor_idx] > 0
            combined_features = (group_features | neighbor_feature).astype(int)

            if len(set(combined_features)) < 2:
                continue

            # Calculate mutual information gain
            try:
                score = mutual_info_classif(
                    combined_features.reshape(-1, 1),
                    y,
                    discrete_features=True,
                    random_state=42,
                )[0]

                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor_idx
            except Exception:
                # Skip this neighbor if mutual info computation fails
                continue

        return best_neighbor, best_score
