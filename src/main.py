import asyncio
import logging
import os
import cv2
from camera import Camera
from config_loader import ConfigLoader
from decision_unit import DecisionUnit
from image_processor import ImageProcessor
from ml_model import MLModel
from storage import Storage
from timestamping import Timestamping
from image_transfer import ImageTransfer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log', filemode='w')

def load_config():
    """Synchrones Laden der Konfigurationsdatei"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    if config is None:
        logging.error("Failed to load configuration.")
        raise Exception("Configuration loading failed.")
    return config

async def main():
    config = load_config()
    drive_uploader = ImageTransfer(
        service_account_file=config['google_drive']['service_account_file'],
        folder_id=config['google_drive']['folder_id']
    )
    ml_model = MLModel(config['model']['path'])
    camera = Camera(config['camera'])
    image_processor = ImageProcessor(config['image_processor']['target_size'])
    decision_unit = DecisionUnit(config)
    timestamping = Timestamping(config)
    storage = Storage(config, drive_uploader)

    # Initialize camera and ML model asynchronously
    await asyncio.gather(camera.setup_camera_async(), ml_model.load_model_async())

    # Start image processing and predictions
    try:
        while True:
            # Capture image synchronously
            image = camera.capture_image()
            # Process image and predict synchronously
            original, processed_image = image_processor.preprocess_image(image)
            prediction_probability = ml_model.predict_object(processed_image)
            object_detected = decision_unit.make_decision(prediction_probability)

            # Timestamp and save image asynchronously, and start stream
            if object_detected:
                timestamped_name = timestamping.add_timestamp("stone.jpg")
                await storage.save_image_async(timestamped_name, original)
                camera.build_stream(original, prediction_probability * 100, object_detected)
            else:
                camera.build_stream(original, prediction_probability * 100, object_detected)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        logging.info("Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())