import logging
from datetime import datetime


class Timestamping:
    def __init__(self, config):
        self.time_format = config['timestamp']['format']

    def add_timestamp(self, image_name):
        try:
            timestamp = datetime.now().strftime(self.time_format)
            name, ext = image_name.rsplit('.', 1)
            timestamped_name = f"{name}_{timestamp}.{ext}"
            logging.info(f"Timestamp added to image name: {timestamped_name}")
            return timestamped_name
        except Exception as e:
            logging.error(f"Failed to add timestamp to image name: {e}")
            return None