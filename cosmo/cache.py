import json
import os
from enum import Enum

class CacheType(Enum):
    VELOCITY = "velocity"
    #POSITION = "position"

class Cache:
    def __init__(self, filepath="data/cache.json"):
        self.filepath = filepath
        folder_path = os.path.dirname(self.filepath)
        
        # Only try to create if a folder path actually exists (handling strictly filenames)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)
        # ------------------------------------

        self.cache = self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_to_disk(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def get_cached_value(self, key, data_type: CacheType):
        if key not in self.cache:
            return None
        return self.cache[key].get(data_type.value)

    def add_cached_value(self, key, data_type: CacheType, value):
        if key not in self.cache:
            self.cache[key] = {}
        
        self.cache[key][data_type.value] = value
        self._save_to_disk()
