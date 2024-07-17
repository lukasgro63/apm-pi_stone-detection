from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.image_processor import ImageProcessor


@pytest.fixture
def image_processor():
    target_size = (150, 150)
    return ImageProcessor(target_size)

@pytest.fixture
def sample_image():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def sample_rgba_image():
    return np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)

def test_preprocess_image(image_processor, sample_image):
    original_image, processed_image = image_processor.preprocess_image(sample_image)
    
    assert np.array_equal(original_image, sample_image)

    assert processed_image.shape == (1, 150, 150, 3)
    assert processed_image.min() >= 0 and processed_image.max() <= 1

def test_preprocess_image_with_rgba(image_processor, sample_rgba_image):
    original_image, processed_image = image_processor.preprocess_image(sample_rgba_image)
    
    assert processed_image.shape[2] == 3

@patch('image_processor.cv2.resize')
def test_preprocess_image_error(mock_resize, image_processor, sample_image):
    mock_resize.side_effect = Exception("Resize error")
    
    original_image, processed_image = image_processor.preprocess_image(sample_image)
    
    assert processed_image is None
    assert np.array_equal(original_image, sample_image)

