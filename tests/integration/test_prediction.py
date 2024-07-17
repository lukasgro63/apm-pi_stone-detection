import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock picamera on non-Raspberry Pi platforms
if not sys.platform.startswith('linux'):
    sys.modules['picamera'] = MagicMock()
    sys.modules['picamera.array'] = MagicMock()

from camera import Camera
from decision_unit import DecisionUnit
from image_processor import ImageProcessor
from ml_model import MLModel
from storage import Storage
from timestamping import Timestamping

TEST_IMAGE_PATHS = ["images/test_img_false_1.jpg", "images/test_img_true.jpg"]
MODEL_PATH = "../../model/stone_detection_model.tflite" 

@pytest.fixture
async def setup_environment():
    with patch('camera.picamera.PiCamera'), patch('camera.picamera.array.PiRGBArray') as mock_array:
        mock_array.return_value.array = MagicMock()
        camera = Camera(resolution=(640, 480), framerate=24)
        camera.capture_image = AsyncMock(side_effect=[
            cv2.imread(TEST_IMAGE_PATHS[0], cv2.IMREAD_COLOR),
            cv2.imread(TEST_IMAGE_PATHS[1], cv2.IMREAD_COLOR)
        ])
        decision_unit = DecisionUnit(threshold=0.5)
        image_processor = ImageProcessor()
        ml_model = MLModel()  # Use the real MLModel
        storage = Storage(storage_dir='test_images/')
        timestamping = Timestamping()

        await ml_model.load_model(MODEL_PATH)

        if not os.path.exists(storage.storage_dir):
            os.makedirs(storage.storage_dir)

        yield camera, decision_unit, image_processor, ml_model, storage, timestamping

@pytest.mark.asyncio
async def test_image_processing_and_storage(setup_environment):
    async for camera, decision_unit, image_processor, ml_model, storage, timestamping in setup_environment:
        for idx in range(2):
            image = await camera.capture_image()
            processed_image = await image_processor.preprocess_image(image)
            prediction = await ml_model.predict(processed_image)
            should_save = decision_unit.should_save_image(prediction)
            if should_save:
                timestamped_name = timestamping.add_timestamp(f"test_image_{idx}.jpg")
                image_path = storage.save_image(timestamped_name, image)
                assert image_path is not None
                assert 'test_images/' in image_path
                print(f"Image saved at: {image_path}")
                assert os.path.isfile(image_path)
                file_size = os.path.getsize(image_path)
                print(f"File size of saved image: {file_size} bytes")
                loaded_image = cv2.imread(image_path)
                if loaded_image is None:
                    print(f"Failed to load image at path: {image_path}")
                assert loaded_image is not None
            else:
                assert not should_save
