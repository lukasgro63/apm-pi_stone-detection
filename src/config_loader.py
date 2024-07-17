import logging
import os

import yaml


class ConfigLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_config(self):
        try:
            with open(self.filepath, 'r') as f:
                config = yaml.safe_load(f)
            logging.debug(f"Config loaded: {config}")
            return config
        except Exception as e:
            logging.error(f"Failed to read configuration file: {e}")
            return None
