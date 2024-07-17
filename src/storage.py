import os
import aiofiles
import cv2
import logging
from image_transfer import ImageTransfer

class Storage:
    def __init__(self, config, drive_uploader):
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.storage_dir = os.path.join(base_path, config['storage']['directory'])
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        self.drive_uploader = drive_uploader

    async def save_image_async(self, image_path, image_data):
        full_path = os.path.join(self.storage_dir, image_path)
        try:
            is_success, buffer = cv2.imencode('.jpg', image_data)
            if not is_success:
                raise ValueError("Failed to encode image")
            
            async with aiofiles.open(full_path, 'wb') as file:
                await file.write(buffer.tobytes())
            logging.info(f"Image successfully saved to {full_path}")

            upload_result = await self.drive_uploader.upload_file(full_path)
            if upload_result:
                logging.info(f"Image {image_path} successfully uploaded to Google Drive.")
            else:
                logging.error(f"Failed to upload {image_path} to Google Drive.")
            return True
        except Exception as e:
            logging.error(f"Failed to save image {full_path}: {e}")
            return False
