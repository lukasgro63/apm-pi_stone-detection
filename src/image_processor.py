import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, target_size):
        self.target_size = tuple(target_size)
        self.executor = ThreadPoolExecutor(max_workers=1)
        logging.info(f"ImageProcessor initialized with target size: {self.target_size}")

    def preprocess_image(self, image):
        try:
            processed_image = self._resize_and_normalize(image)
            logging.debug("Image successfully preprocessed.")
            return (image, processed_image)
        except Exception as e:
            logging.error(f"Failed to preprocess image: {e}")
            return (image, None)

    def _resize_and_normalize(self, image):
        """
        Resizes and normalizes the image to the target size and scale [0, 1].
        Converts the image to 3 channels if it has 4 channels (RGBA to RGB).
        """
        resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Prüfen, ob das Bild 4 Kanäle hat und auf 3 Kanäle reduzieren
        if resized_image.shape[2] == 4:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2RGB)
        normalized_image = resized_image.astype('float32') / 255.0

        # Erweitern der Dimensionen, um der Eingabeerwartung des Modells zu entsprechen
        normalized_image = np.expand_dims(normalized_image, axis=0)
        return normalized_image
