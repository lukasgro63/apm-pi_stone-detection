import os
from unittest.mock import mock_open, patch

import cv2
import numpy as np
import pytest

from src.storage import Storage


class TestStorage:
    @pytest.fixture
    def storage(self):
        return Storage()

    @patch('os.makedirs')
    @patch('cv2.imwrite')
    def test_save_image(self, mock_imwrite, mock_makedirs, storage):
        """Testet das Speichern von Bildern."""
        image_path = 'test_image.jpg'
        image_data = np.zeros((100, 100, 3), dtype=np.uint8)

        saved_path = storage.save_image(image_path, image_data)

        full_path = os.path.join(storage.storage_dir, image_path)
        mock_imwrite.assert_called_once_with(full_path, image_data)
        assert saved_path == full_path

    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    def test_storage_initialization(self, mock_makedirs, mock_exists):
        """Testet die Initialisierung des Storage-Verzeichnisses."""
        storage_dir = 'test_images'
        storage = Storage(storage_dir=storage_dir)

        mock_exists.assert_called_once_with(storage_dir)
        mock_makedirs.assert_called_once_with(storage_dir)
