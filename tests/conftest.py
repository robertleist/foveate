import numpy as np
import pytest

from foveate import MockBackbone


@pytest.fixture
def backbone():
    return MockBackbone(image_size=224, patch_size=14)


@pytest.fixture
def two_squares():
    """128x128 image: two red squares (target) + one blue square (distractor), + exemplar mask."""
    img = np.zeros((128, 128, 3), np.uint8)
    img[20:40, 20:40] = (220, 30, 30)
    img[80:100, 80:100] = (220, 30, 30)
    img[20:40, 80:100] = (30, 30, 220)
    ex = np.zeros((128, 128), np.uint8)
    ex[20:40, 20:40] = 1
    return img, [ex]
