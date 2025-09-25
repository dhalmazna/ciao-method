"""
Segmentation components for CIAO implementation.
Provides superpixel and hexagonal grid segmentation with adjacency graphs.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

import cv2
import networkx as nx
import numpy as np
from skimage.segmentation import felzenszwalb, slic

from ciao.typing import Image, SegmentMask


class BaseSegmentation(ABC):
    """Abstract base class for segmentation methods."""

    @abstractmethod
    def segment(self, image: Image) -> Tuple[SegmentMask, nx.Graph]:
        """
        Segment image into basic segments and create adjacency graph.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            segments: Label array where each pixel has segment ID
            adjacency_graph: NetworkX graph representing segment adjacency
        """
        pass

    def _build_adjacency_graph(self, segments: SegmentMask) -> nx.Graph:
        """Build adjacency graph from segment labels."""
        adj_graph = nx.Graph()

        # Get unique segment IDs
        segment_ids = np.unique(segments)
        adj_graph.add_nodes_from(segment_ids)

        # Find adjacent segments by checking neighboring pixels
        height, width = segments.shape

        # Check horizontal adjacency
        for y in range(height):
            for x in range(width - 1):
                seg1, seg2 = segments[y, x], segments[y, x + 1]
                if seg1 != seg2:
                    adj_graph.add_edge(seg1, seg2)

        # Check vertical adjacency
        for y in range(height - 1):
            for x in range(width):
                seg1, seg2 = segments[y, x], segments[y + 1, x]
                if seg1 != seg2:
                    adj_graph.add_edge(seg1, seg2)

        return adj_graph


class SuperpixelSegmentation(BaseSegmentation):
    """Superpixel segmentation using SLIC algorithm."""

    def __init__(
        self, n_segments: int = 100, compactness: float = 10.0, algorithm: str = "slic"
    ):
        """
        Initialize superpixel segmentation.

        Args:
            n_segments: Approximate number of segments
            compactness: Balance color proximity vs space proximity
            algorithm: 'slic' or 'felzenszwalb'
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.algorithm = algorithm

    def segment(self, image: Image) -> Tuple[SegmentMask, nx.Graph]:
        """Segment image using superpixels."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        if self.algorithm == "slic":
            segments = slic(
                image, n_segments=self.n_segments, compactness=self.compactness, sigma=1
            )
        elif self.algorithm == "felzenszwalb":
            segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        adjacency_graph = self._build_adjacency_graph(segments)
        return segments, adjacency_graph


class HexagonalSegmentation(BaseSegmentation):
    """Hexagonal grid segmentation."""

    def __init__(self, hex_size: int = 20):
        """
        Initialize hexagonal segmentation.

        Args:
            hex_size: Size of hexagonal segments in pixels
        """
        self.hex_size = hex_size

    def segment(self, image: Image) -> Tuple[SegmentMask, nx.Graph]:
        """Segment image using hexagonal grid."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        height, width = image.shape[:2]
        segments = np.zeros((height, width), dtype=np.int32)

        # Hexagonal grid parameters
        hex_width = self.hex_size
        hex_height = int(self.hex_size * np.sqrt(3) / 2)

        segment_id = 0
        segment_centers = {}

        # Create hexagonal grid
        for row in range(0, height, hex_height):
            for col in range(0, width, hex_width):
                # Offset every other row for hexagonal pattern
                offset = (hex_width // 2) if (row // hex_height) % 2 == 1 else 0
                actual_col = col + offset

                if actual_col >= width:
                    continue

                # Create hexagonal mask
                hex_mask = self._create_hexagon_mask(
                    height, width, actual_col, row, self.hex_size
                )

                segments[hex_mask] = segment_id
                segment_centers[segment_id] = (
                    row + hex_height // 2,
                    actual_col + hex_width // 2,
                )
                segment_id += 1

        # Build adjacency graph based on spatial proximity
        adjacency_graph = self._build_hex_adjacency_graph(segments, segment_centers)

        return segments, adjacency_graph

    def _create_hexagon_mask(
        self, height: int, width: int, center_x: int, center_y: int, size: int
    ) -> np.ndarray:
        """Create hexagonal mask around center point."""
        y, x = np.ogrid[:height, :width]

        # Hexagon vertices
        angles = np.linspace(0, 2 * np.pi, 7)  # 6 vertices + closing
        hex_x = center_x + size * np.cos(angles)
        hex_y = center_y + size * np.sin(angles)

        # Create polygon mask
        vertices = np.column_stack([hex_x, hex_y]).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 1)

        return mask.astype(bool)

    def _build_hex_adjacency_graph(
        self, segments: SegmentMask, centers: Dict[int, Tuple[int, int]]
    ) -> nx.Graph:
        """Build adjacency graph for hexagonal segments."""
        adjacency_graph = nx.Graph()
        adjacency_graph.add_nodes_from(centers.keys())

        # Connect segments that are spatially adjacent
        for seg_id, (y1, x1) in centers.items():
            for other_id, (y2, x2) in centers.items():
                if seg_id >= other_id:
                    continue

                # Check if segments are adjacent (within threshold distance)
                distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if distance <= self.hex_size * 1.5:  # Threshold for adjacency
                    adjacency_graph.add_edge(seg_id, other_id)

        return adjacency_graph


class SegmentAnalyzer:
    """Utility class for analyzing segments."""

    @staticmethod
    def get_segment_pixels(segments: SegmentMask, segment_id: int):
        """Get pixel coordinates for a segment."""
        return np.where(segments == segment_id)

    @staticmethod
    def get_segment_mask(segments: SegmentMask, segment_ids: List[int]) -> np.ndarray:
        """Get binary mask for multiple segments."""
        mask = np.zeros_like(segments, dtype=bool)
        for seg_id in segment_ids:
            mask |= segments == seg_id
        return mask

    @staticmethod
    def get_connected_components(
        graph: nx.Graph, segment_ids: Set[int]
    ) -> List[Set[int]]:
        """Get connected components from a subgraph."""
        subgraph = graph.subgraph(segment_ids)
        return [set(component) for component in nx.connected_components(subgraph)]

    @staticmethod
    def compute_segment_statistics(
        image: Image, segments: SegmentMask, segment_id: int
    ) -> Dict:
        """Compute statistics for a segment."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        mask = segments == segment_id
        segment_pixels = image[mask]

        if len(segment_pixels) == 0:
            return {}

        return {
            "mean_color": np.mean(segment_pixels, axis=0),
            "std_color": np.std(segment_pixels, axis=0),
            "size": len(segment_pixels),
            "centroid": np.mean(np.where(mask), axis=1),
        }
