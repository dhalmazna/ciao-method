"""
Obfuscation components for CIAO implementation.
Provides pixel interlacing, zeroing out, noise, and blur obfuscation techniques.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import cv2
import numpy as np

from ciao.typing import Image, SegmentMask


class BaseObfuscation(ABC):
    """Abstract base class for obfuscation methods."""

    @abstractmethod
    def obfuscate_image(self, image: Image) -> Image:
        """
        Create obfuscated version of the entire image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Obfuscated image
        """
        pass

    @abstractmethod
    def obfuscate_segments(
        self, image: Image, segments: SegmentMask, segment_ids: List[int]
    ) -> Image:
        """
        Obfuscate specific segments of the image.

        Args:
            image: Original image
            segments: Segment label array
            segment_ids: List of segment IDs to obfuscate

        Returns:
            Image with specified segments obfuscated
        """
        pass


class PixelInterlacing(BaseObfuscation):
    """
    Pixel interlacing obfuscation as described in the CIAO paper.
    Divides image into n×n squares and applies horizontal/vertical flipping.
    """

    def __init__(self, grid_size: int = 8):
        """
        Initialize pixel interlacing obfuscation.

        Args:
            grid_size: Size of the grid squares (n×n)
        """
        self.grid_size = grid_size

    def obfuscate_image(self, image: Image) -> Image:
        """Create obfuscated version using pixel interlacing."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        obfuscated = image.copy()
        height, width = image.shape[:2]

        # Process each grid square
        for y in range(0, height, self.grid_size):
            for x in range(0, width, self.grid_size):
                y_end = min(y + self.grid_size, height)
                x_end = min(x + self.grid_size, width)

                # Extract square
                square = image[y:y_end, x:x_end]

                # Apply horizontal interlacing (every second column)
                interlaced_square = square.copy()
                for col in range(1, square.shape[1], 2):
                    # Flip horizontally and take every second column
                    flipped = np.fliplr(square)
                    interlaced_square[:, col] = flipped[:, col]

                # Apply vertical interlacing (every second row)
                for row in range(1, interlaced_square.shape[0], 2):
                    # Flip vertically and take every second row
                    flipped = np.flipud(interlaced_square)
                    interlaced_square[row, :] = flipped[row, :]

                obfuscated[y:y_end, x:x_end] = interlaced_square

        return obfuscated

    def obfuscate_segments(
        self, image: Image, segments: SegmentMask, segment_ids: List[int]
    ) -> Image:
        """Obfuscate specific segments using pixel interlacing."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        # First create fully obfuscated image
        obfuscated_full = self.obfuscate_image(image)

        # Create result starting with original image
        result = image.copy()

        # Replace only the specified segments
        for segment_id in segment_ids:
            mask = segments == segment_id
            result[mask] = obfuscated_full[mask]

        return result


class ZeroOut(BaseObfuscation):
    """Simple zeroing out obfuscation - sets pixels to zero."""

    def __init__(self, value: Union[int, float, List] = 0):
        """
        Initialize zero-out obfuscation.

        Args:
            value: Value to set pixels to (scalar or per-channel list)
        """
        self.value = value

    def obfuscate_image(self, image: Image) -> Image:
        """Create obfuscated version by zeroing out."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        obfuscated = np.full_like(image, self.value)
        return obfuscated

    def obfuscate_segments(
        self, image: Image, segments: SegmentMask, segment_ids: List[int]
    ) -> Image:
        """Obfuscate specific segments by zeroing out."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        result = image.copy()

        for segment_id in segment_ids:
            mask = segments == segment_id
            result[mask] = self.value

        return result


class NoiseObfuscation(BaseObfuscation):
    """Noise-based obfuscation using Gaussian or uniform noise."""

    def __init__(
        self,
        noise_type: str = "gaussian",
        intensity: float = 0.3,
        mean: float = 0.5,
        std: float = 0.1,
    ):
        """
        Initialize noise obfuscation.

        Args:
            noise_type: 'gaussian' or 'uniform'
            intensity: Noise intensity (0-1)
            mean: Mean for gaussian noise or center for uniform
            std: Standard deviation for gaussian or range for uniform
        """
        self.noise_type = noise_type
        self.intensity = intensity
        self.mean = mean
        self.std = std

    def _generate_noise(self, shape: tuple) -> np.ndarray:
        """Generate noise array with specified properties."""
        if self.noise_type == "gaussian":
            noise = np.random.normal(self.mean, self.std, shape)
        elif self.noise_type == "uniform":
            low = self.mean - self.std
            high = self.mean + self.std
            noise = np.random.uniform(low, high, shape)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        # Clip to valid range [0, 1]
        noise = np.clip(noise, 0, 1)
        return noise

    def obfuscate_image(self, image: Image) -> Image:
        """Create obfuscated version using noise."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        # Normalize image to [0, 1] if needed
        if image.max() > 1:
            normalized_image = image.astype(np.float32) / 255.0
        else:
            normalized_image = image.astype(np.float32)

        # Generate noise
        noise = self._generate_noise(normalized_image.shape)

        # Blend original with noise
        obfuscated = (1 - self.intensity) * normalized_image + self.intensity * noise

        # Convert back to original range
        if image.max() > 1:
            obfuscated = (obfuscated * 255).astype(image.dtype)
        else:
            obfuscated = obfuscated.astype(image.dtype)

        return obfuscated

    def obfuscate_segments(
        self, image: Image, segments: SegmentMask, segment_ids: List[int]
    ) -> Image:
        """Obfuscate specific segments using noise."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        result = image.copy()

        # Normalize if needed
        if image.max() > 1:
            normalized_result = result.astype(np.float32) / 255.0
        else:
            normalized_result = result.astype(np.float32)

        for segment_id in segment_ids:
            mask = segments == segment_id
            if np.any(mask):
                # Generate noise for this segment
                segment_shape = normalized_result[mask].shape
                noise = self._generate_noise(segment_shape)

                # Apply noise to segment
                normalized_result[mask] = (1 - self.intensity) * normalized_result[
                    mask
                ] + self.intensity * noise

        # Convert back to original range
        if image.max() > 1:
            result = (normalized_result * 255).astype(image.dtype)
        else:
            result = normalized_result.astype(image.dtype)

        return result


class BlurObfuscation(BaseObfuscation):
    """Blur-based obfuscation using Gaussian blur."""

    def __init__(self, kernel_size: int = 15, sigma: float = 5.0):
        """
        Initialize blur obfuscation.

        Args:
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation for Gaussian kernel
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma

    def obfuscate_image(self, image: Image) -> Image:
        """Create obfuscated version using Gaussian blur."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        obfuscated = cv2.GaussianBlur(
            image, (self.kernel_size, self.kernel_size), self.sigma
        )
        return obfuscated

    def obfuscate_segments(
        self, image: Image, segments: SegmentMask, segment_ids: List[int]
    ) -> Image:
        """Obfuscate specific segments using blur."""
        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        # Create fully blurred image
        blurred_full = self.obfuscate_image(image)

        # Start with original image
        result = image.copy()

        # Replace only specified segments
        for segment_id in segment_ids:
            mask = segments == segment_id
            result[mask] = blurred_full[mask]

        return result


class ObfuscationComparator:
    """Utility class for comparing different obfuscation methods."""

    @staticmethod
    def compare_methods(
        image: Image, segments: SegmentMask, segment_ids: List[int]
    ) -> dict:
        """
        Compare different obfuscation methods on the same segments.

        Args:
            image: Original image
            segments: Segment labels
            segment_ids: Segments to obfuscate

        Returns:
            Dictionary mapping method names to obfuscated images
        """
        methods = {
            "interlacing": PixelInterlacing(),
            "zero": ZeroOut(),
            "noise": NoiseObfuscation(),
            "blur": BlurObfuscation(),
        }

        results = {}
        for name, obfuscator in methods.items():
            results[name] = obfuscator.obfuscate_segments(image, segments, segment_ids)

        return results

    @staticmethod
    def visualize_comparison(
        image: Image, obfuscated_images: dict, save_path: str = None
    ):
        """
        Create side-by-side visualization of obfuscation methods.

        Args:
            image: Original image
            obfuscated_images: Dict from compare_methods
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt

        # Convert to numpy if tensor
        if hasattr(image, "numpy"):
            image = image.numpy()

        n_methods = len(obfuscated_images)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

        # Show original
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Show obfuscated versions
        for i, (method, obf_image) in enumerate(obfuscated_images.items()):
            axes[i + 1].imshow(obf_image)
            axes[i + 1].set_title(f"{method.capitalize()}")
            axes[i + 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
