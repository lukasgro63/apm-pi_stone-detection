import sys
from unittest.mock import MagicMock, patch

import pytest

if not sys.platform.startswith('linux'):
    sys.modules['picamera2'] = MagicMock()

from src.camera import Camera


@pytest.fixture
def mock_picamera2(mocker):
    # Mock the Picamera2 class
    with patch('camera.Picamera2') as mock:
        yield mock

class TestCamera:
    @pytest.fixture
    def camera(self, mock_picamera2):
        config = {'resolution': (1920, 1080), 'framerate': 30}
        return Camera(config)

    @pytest.mark.asyncio
    async def test_setup_camera_async(self, camera, mock_picamera2):
        await camera.setup_camera_async()
        mock_picamera2.return_value.create_preview_configuration.assert_called_once()
        mock_picamera2.return_value.configure.assert_called_once()
        mock_picamera2.return_value.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_image(self, camera, mock_picamera2):
        mock_picamera2.return_value.capture_array.return_value = 'fake_image'
        image = await camera.capture_image()
        assert image == 'fake_image'
        mock_picamera2.return_value.capture_array.assert_called_once()

    def test_build_stream(self, camera, mock_picamera2):
        processed_image = 'processed_image_mock'
        stone_prediction = 0.5
        stone_detected = True
        with patch('camera.cv2.putText'), patch('camera.cv2.imshow'), patch('camera.cv2.resizeWindow'):
            camera.build_stream(processed_image, stone_prediction, stone_detected)
            camera.cv2.putText.assert_called()
            camera.cv2.imshow.assert_called()
            camera.cv2.resizeWindow.assert_called()
