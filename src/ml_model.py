import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tflite_runtime.interpreter as tflite


class MLModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.executor = ThreadPoolExecutor(max_workers=1)  # Erhöht die Anzahl der Threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    async def load_model_async(self):
        """Loads the TensorFlow Lite model from the path in the config.yaml and allocates tensors."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self.executor, self._load_model)
            logging.info(f"Model loaded and tensors allocated from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load and allocate model: {e}")

    def _load_model(self):
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e

    def predict_object(self, image):
        try:
            prediction = self._predict(image)
            logging.info(f"Prediction made with probability: {prediction}")
            return prediction
        except Exception as e:
            logging.error(f"Failed to make prediction: {e}")
            return None

    def _predict(self, image):
        # Ensure the image matches the input requirements
        if image.shape != tuple(self.input_details[0]['shape']):
            raise ValueError(f"Expected image shape {tuple(self.input_details[0]['shape'])}, but got {image.shape}")
        
        # Startzeit messen
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        prediction = output_data[0][0]

        # Endzeit messen und Dauer berechnen
        elapsed_time = time.time() - start_time
        logging.info(f"Prediction made in {elapsed_time:.3f} seconds")

        return prediction
