from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from image_transfer import ImageTransfer


@pytest.fixture
def setup_image_transfer():
    with patch('google.oauth2.service_account.Credentials.from_service_account_file', return_value=MagicMock()), \
         patch('googleapiclient.discovery.build', return_value=MagicMock()) as mock_build:
        service = mock_build.return_value
        service.files.return_value.create.return_value.execute.return_value = {'id': '12345'}
        return ImageTransfer("dummy_service_account.json", "dummy_folder_id")

@pytest.mark.asyncio
async def test_upload_file_single(setup_image_transfer):
    image_transfer = setup_image_transfer
    file_path = "test.jpg"
    
    with patch.object(image_transfer, '_upload_file', new_callable=AsyncMock) as mock_upload:
        mock_upload.return_value = True
        await image_transfer.upload_file(file_path)
        mock_upload.assert_called_once_with(file_path)
        assert file_path not in image_transfer.upload_queue

@pytest.mark.asyncio
async def test_process_queue_multiple(setup_image_transfer):
    image_transfer = setup_image_transfer
    files_to_upload = ["test1.jpg", "test2.jpg", "test3.jpg"]
    image_transfer.upload_queue.extend(files_to_upload)
    
    with patch.object(image_transfer, '__execute_upload', return_value=True) as mock_execute_upload:
        mock_execute_upload.return_value = True
        await image_transfer.process_queue()
        assert len(image_transfer.upload_queue) == 0

@pytest.mark.asyncio
async def test_upload_file_failure_and_retry(setup_image_transfer):
    image_transfer = setup_image_transfer
    file_path = "failed_upload.jpg"
    
    with patch.object(image_transfer, '__execute_upload', side_effect=[False, True]) as mock_execute:
        await image_transfer.upload_file(file_path)
        assert mock_execute.call_count == 2

