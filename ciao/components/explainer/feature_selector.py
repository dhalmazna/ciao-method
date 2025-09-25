"""
Main CIAO explainer implementation using Lightning framework.
"""

import logging
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

from ciao.components.obfuscation.base import BaseObfuscation
from ciao.typing import Classifier, Image, SegmentMask

logger = logging.getLogger(__name__)


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
        use_fallback_groups: bool = True,
        max_group_size_ratio: float = 0.05,
    ):
        """Initialize CIAO feature selector."""
        self.model = model
        self.target_class = target_class
        self.base_eta = eta
        self.eta = eta
        self.radius_range = radius_range
        self.adaptive_eta = adaptive_eta
        self.use_fallback_groups = use_fallback_groups
        self.max_group_size_ratio = max_group_size_ratio
        self.model.eval()

        # Log model info for debugging
        try:
            if hasattr(model, "backbone") and hasattr(getattr(model, "backbone"), "fc"):
                backbone = getattr(model, "backbone")
                fc_layer = getattr(backbone, "fc")
                fc_weight = fc_layer.weight.data
                fc_bias = fc_layer.bias.data if fc_layer.bias is not None else None
                logger.info(
                    f"Model final layer weights: mean={fc_weight.mean():.4f}, std={fc_weight.std():.4f}"
                )
                if fc_bias is not None:
                    logger.info(f"Model final layer bias: {fc_bias.item():.4f}")
        except Exception as e:
            logger.warning(f"Could not inspect model parameters: {e}")

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
            


            # Handle both single value and multi-class predictions
            if original_pred.numel() == 1:
                # Single value prediction (binary classification sigmoid output)
                orig_score = original_pred.item()
                obfus_score = obfuscated_pred.item()
            else:
                # Multi-class prediction (including our converted 2-class binary)
                # Handle case where batch dimension might still be present
                if len(original_pred.shape) > 1:
                    orig_score = original_pred[0, target_idx].item()  # [batch, class]
                    obfus_score = obfuscated_pred[0, target_idx].item()
                else:
                    orig_score = original_pred[target_idx].item()  # [class]
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

        # Log surrogate dataset statistics
        logger.info(
            f"Surrogate dataset: {len(local_groups)} local groups, "
            f"delta_threshold={delta_threshold:.4f}, "
            f"delta_mean={np.mean(deltas):.4f}, "
            f"delta_std={np.std(deltas):.4f}, "
            f"positive_ratio={np.mean(y_array):.4f}"
        )

        # Adapt eta if enabled
        if self.adaptive_eta:
            old_eta = self.eta
            self.eta = self._compute_adaptive_eta(X_array, y_array, deltas)
            logger.info(f"Adaptive eta: {old_eta:.4f} -> {self.eta:.4f}")

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

        # Special handling for very small deltas (high-confidence predictions)
        if delta_mean < 0.01:  # Very small changes in prediction
            logger.info(f"Small delta_mean ({delta_mean:.4f}), using conservative eta")
            # Use much smaller eta for high-confidence cases
            conservative_eta = min(self.base_eta * 0.1, 0.002)
            return float(conservative_eta)

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
        # Lower minimum threshold for better sensitivity
        adaptive_eta = max(0.001, min(adaptive_eta, self.base_eta))

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
                logger.warning(
                    f"No seed feature found for remaining {len(remaining_features)} features"
                )
                break

            current_group = {seed_idx}
            remaining_features.remove(seed_idx)
            logger.debug(f"Starting new group with seed {seed_idx}")

            # Grow group iteratively with maximum size limit
            iteration = 0
            max_group_size = max(3, int(len(segment_ids) * self.max_group_size_ratio))  # Configurable max ratio
            while True:
                iteration += 1
                
                # Stop if group is getting too large
                if len(current_group) >= max_group_size:
                    logger.debug(
                        f"Group {len(feature_groups)}: Reached max size {max_group_size} (iteration {iteration})"
                    )
                    break
                
                neighbors = self._get_group_neighbors(
                    current_group, segment_ids, adjacency_graph, remaining_features
                )

                if not neighbors:
                    logger.debug(
                        f"Group {len(feature_groups)}: No neighbors found (iteration {iteration})"
                    )
                    break

                best_neighbor, best_score = self._find_best_neighbor(
                    X, y, current_group, neighbors
                )

                if best_neighbor is None:
                    logger.debug(
                        f"Group {len(feature_groups)}: No valid neighbor found (iteration {iteration})"
                    )
                    break

                if best_score < self.eta:
                    logger.debug(
                        f"Group {len(feature_groups)}: Best score {best_score:.4f} < eta {self.eta:.4f} (iteration {iteration})"
                    )
                    break

                logger.debug(
                    f"Group {len(feature_groups)}: Adding neighbor {best_neighbor} with score {best_score:.4f}"
                )
                current_group.add(best_neighbor)
                remaining_features.remove(best_neighbor)

            # Convert indices back to segment IDs
            segment_group = {segment_ids[idx] for idx in current_group}
            feature_groups.append(segment_group)
            logger.info(
                f"Created group {len(feature_groups)} with {len(segment_group)} segments"
            )

        # Fallback strategy: if no groups were formed, create at least one group
        # with the most informative segments
        if len(feature_groups) == 0 and self.use_fallback_groups:
            logger.warning("No groups formed, applying fallback strategy")
            fallback_group = self._create_fallback_group(
                X, y, segment_ids, adjacency_graph
            )
            if fallback_group:
                feature_groups.append(fallback_group)
                logger.info(
                    f"Created fallback group with {len(fallback_group)} segments"
                )

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

        # Ensure tensor is float32 (ResNet expects float, not uint8)
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()

        # Normalize to [0, 1] range if values are in [0, 255] range
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

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
            raw_prediction = self.model(image_tensor)
            logger.debug(
                f"Raw model output shape: {raw_prediction.shape}, values: {raw_prediction}"
            )

            if len(raw_prediction.shape) > 1 and raw_prediction.shape[1] > 1:
                # Multi-class classification
                prediction = torch.softmax(raw_prediction, dim=1)
                logger.debug(f"Multi-class probabilities: {prediction}")
            else:
                # Binary classification - convert single sigmoid output to 2-class probabilities
                sigmoid_output = torch.sigmoid(raw_prediction)  # Keep batch dimension for now
                # Create 2-class probability distribution: [P(class=0), P(class=1)]
                prediction = torch.cat([1 - sigmoid_output, sigmoid_output], dim=1)
                logger.debug(
                    f"Binary probabilities: {prediction} (from sigmoid: {sigmoid_output})"
                )

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

    def _create_fallback_group(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segment_ids: np.ndarray,
        adjacency_graph: nx.Graph,
    ) -> Optional[Set[int]]:
        """Create a fallback group when no groups are formed using standard algorithm."""
        if len(segment_ids) == 0:
            return None

        # Find segments with highest individual mutual information
        individual_scores = []
        valid_features = []

        for i in range(X.shape[1]):
            if len(set(X[:, i])) < 2:  # Skip features with no variation
                continue

            try:
                score = mutual_info_classif(
                    X[:, [i]], y, discrete_features=True, random_state=42
                )[0]
                individual_scores.append((i, score))
                valid_features.append(i)
            except Exception:
                continue

        if not individual_scores:
            # If no valid features, just take the first few segments
            logger.warning("No valid features for fallback, using first segments")
            fallback_size = min(3, len(segment_ids))
            return {int(segment_ids[i]) for i in range(fallback_size)}

        # Sort by score and take top segments
        individual_scores.sort(key=lambda x: x[1], reverse=True)

        # Start with best segment and grow using adjacency
        fallback_group = set()
        seed_feature_idx = individual_scores[0][0]
        seed_segment_id = int(segment_ids[seed_feature_idx])
        fallback_group.add(seed_segment_id)

        # Add adjacent segments up to a reasonable size
        max_fallback_size = max(3, int(len(segment_ids) * self.max_group_size_ratio * 2))  # Slightly larger for fallback
        added_segments = {seed_segment_id}

        for _ in range(max_fallback_size - 1):
            candidates = set()
            for seg_id in added_segments:
                if seg_id in adjacency_graph:
                    for neighbor in adjacency_graph.neighbors(seg_id):
                        if neighbor not in added_segments:
                            candidates.add(neighbor)

            if not candidates:
                break

            # Add the candidate with highest individual score
            best_candidate = None
            best_score = -1

            for candidate in candidates:
                try:
                    candidate_idx = np.where(segment_ids == candidate)[0][0]
                    if candidate_idx < len(individual_scores):
                        # Find score for this candidate
                        for idx, score in individual_scores:
                            if idx == candidate_idx:
                                if score > best_score:
                                    best_score = score
                                    best_candidate = candidate
                                break
                except (IndexError, ValueError):
                    continue

            if best_candidate is not None:
                fallback_group.add(int(best_candidate))
                added_segments.add(int(best_candidate))
            else:
                break

        logger.info(f"Created fallback group with {len(fallback_group)} segments")
        return fallback_group
