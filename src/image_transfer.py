import asyncio
import os
import logging
from collections import deque
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class ImageTransfer:
    def __init__(self, service_account_file, folder_id):
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/drive.file'])
        self.service = build('drive', 'v3', credentials=credentials)
        self.folder_id = folder_id
        self.upload_queue = deque()
        self.is_uploading = False

    async def upload_file(self, file_path):
        self.upload_queue.append(file_path)
        if not self.is_uploading:
            self.is_uploading = True
            await self.process_queue()

    async def process_queue(self):
        while self.upload_queue:
            file_path = self.upload_queue.popleft()
            result = await self._upload_file(file_path)
            if not result:
                logging.error(f"Retry logic for failed upload of {file_path}")
                #TODO: Logik zum erneuten Versuch hinzugefügt werden
        self.is_uploading = False

    async def _upload_file(self, file_path):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.__execute_upload, file_path)

    def __execute_upload(self, file_path):
        file_metadata = {'name': os.path.basename(file_path), 'parents': [self.folder_id]}
        media = MediaFileUpload(file_path, mimetype='image/jpeg')
        try:
            file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logging.info(f'File ID: {file["id"]} uploaded to Google Drive')
            return True
        except Exception as e:
            logging.error(f'Failed to upload file to Google Drive: {e}')
            return False
