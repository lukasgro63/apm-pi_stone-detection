from datetime import datetime

import pytest

from src.timestamping import Timestamping


class TestTimestamping:
    @pytest.fixture
    def timestamping(self):
        return Timestamping()

    def test_add_timestamp(self, timestamping):
        test_image_name = "test_image.jpg"
        timestamped_name = timestamping.add_timestamp(test_image_name)
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        assert current_time in timestamped_name
        assert test_image_name.split('.')[0] in timestamped_name
        assert test_image_name.split('.')[1] in timestamped_name

    def test_add_timestamp_format(self, timestamping):
        test_image_name = "test_image.png"
        timestamped_name = timestamping.add_timestamp(test_image_name)
        
        assert timestamped_name.endswith(".png")
