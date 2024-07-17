import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
from picamera2 import Picamera2


class Camera:
    def __init__(self, config):
        self.config = config
        self.picam2 = Picamera2()
        self.executor = ThreadPoolExecutor(max_workers=1)
        logging.info("Camera initialized with resolution %s and framerate %s", config['resolution'], config['framerate'])

    async def setup_camera_async(self):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self.executor, self._setup_camera)
            logging.info("Camera configured with resolution: %s", tuple(self.config['resolution']))
        except Exception as e:
            logging.error(f"Failed to configure camera: {e}")

    def _setup_camera(self):
        camera_config = self.picam2.create_preview_configuration(main={"size": tuple(self.config['resolution']), "format": 'XRGB8888'})
        self.picam2.configure(camera_config)
        self.picam2.start()

    def capture_image(self):
        """Capture an image from the camera and log the process."""
        try:
            image = self._capture_image()
            logging.debug("Image captured")
            return image
        except Exception as e:
            logging.error(f"Failed to capture image: {e}")
            return None

    def _capture_image(self):
        image = self.picam2.capture_array()
        return image

    def build_stream(self, processed_image, stone_prediction=0, stone_detected=False):
        """Build the live stream with overlays."""
        try:
            self._build_stream(processed_image, stone_prediction, stone_detected)
            logging.debug("Stream built")
        except Exception as e:
            logging.error(f"Failed to build stream: {e}")

    def _build_stream(self, processed_image, stone_prediction=0, stone_detected=False):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(processed_image, f"Stone Prediction: {stone_prediction}%", (10, 30), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(processed_image, f"Stone Detected: {stone_detected}", (10, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Live Camera Stream', processed_image)
        cv2.resizeWindow('Live Camera Stream', 960, 540)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Stream terminated by user.")
