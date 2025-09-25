"""
Simple test script to validate CIAO migration.
"""

from pathlib import Path

import numpy as np
import torch

# Test imports
try:
    from ciao.components.explainer import CIAOExplainer
    from ciao.components.obfuscation import BlurObfuscation
    from ciao.components.segmentation import SuperpixelSegmentation

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def test_basic_functionality():
    """Test basic CIAO functionality with synthetic data."""
    print("Testing basic CIAO functionality...")

    # Create synthetic image
    image = np.random.rand(224, 224, 3) * 255
    image = image.astype(np.uint8)

    # Test segmentation
    segmenter = SuperpixelSegmentation(n_segments=50)
    segments, adjacency_graph = segmenter.segment(image)
    print(f"✅ Segmentation: {len(np.unique(segments))} segments")

    # Test obfuscation
    obfuscator = BlurObfuscation()
    obfuscated = obfuscator.obfuscate_image(image)
    print("✅ Obfuscation works")

    # Test with mock classifier
    class MockClassifier:
        def __call__(self, x):
            # Return random predictions
            batch_size = x.shape[0]
            return torch.rand(batch_size, 10)  # 10 classes

        def eval(self):
            pass

        def parameters(self):
            return [torch.tensor([1.0])]  # Mock parameter for device detection

    mock_classifier = MockClassifier()

    # Test explainer (without running full explanation)
    explainer = CIAOExplainer(
        classifier=mock_classifier, segmenter=segmenter, obfuscator=obfuscator
    )
    print("✅ CIAO explainer created successfully")

    print("🎉 All basic tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
