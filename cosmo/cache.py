import csv
import json
import os
from enum import Enum
import dataclasses

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

class CacheType(Enum):
    VELOCITY = "velocity"
    METRICS = "metrics"
    RESULTS = "results"

class CacheFormat(Enum):
    JSON = "json"
    CSV = "csv"

class Cache:
    def __init__(self, name="cache", format: CacheFormat = CacheFormat.CSV, _data_dir="data"):
        self.format = format
        self.name = name
        self._data_dir = _data_dir
        extension = "." + format.value
        self.filepath = os.path.join(_data_dir, name + extension)
        self.changes = 0
        folder_path = os.path.dirname(self.filepath)

        # Only try to create if a folder path actually exists (handling strictly filenames)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)
        # ------------------------------------

        self.cache = self._load_from_disk()

    def _alternate_filepath(self):
        alt_ext = ".csv" if self.format == CacheFormat.JSON else ".json"
        return os.path.join(self._data_dir, self.name + alt_ext)

    def _load_from_disk(self):
        if os.path.exists(self.filepath):
            try:
                if self.format == CacheFormat.JSON:
                    return self._load_json(self.filepath)
                else:
                    return self._load_csv(self.filepath)
            except (json.JSONDecodeError, IOError, ValueError):
                return {}

        # Fallback: try alternate format
        alternate_path = self._alternate_filepath()
        if os.path.exists(alternate_path):
            try:
                if self.format == CacheFormat.JSON:
                    return self._load_csv(alternate_path)
                else:
                    return self._load_json(alternate_path)
            except (json.JSONDecodeError, IOError, ValueError):
                pass

        return {}

    def _load_json(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def _load_csv(self, filepath):
        cache = {}
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            try:
                for row in reader:
                    key = row['key']
                    data_type = row['data_type']
                    json_value = row['json_value']
                    value = json.loads(json_value)
                    if key not in cache:
                        cache[key] = {}
                    cache[key][data_type] = value
            except (KeyError, json.JSONDecodeError):
                return {}
        return cache

    def _save_to_disk(self):
        if self.format == CacheFormat.JSON:
            self._save_json()
        else:
            self._save_csv()

    def _save_json(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.cache, f, indent=4, cls=EnhancedJSONEncoder)

    def _save_csv(self):
        rows = []
        for key, data_types in self.cache.items():
            for data_type, value in data_types.items():
                json_value = json.dumps(value, cls=EnhancedJSONEncoder)
                rows.append({
                    'key': key,
                    'data_type': data_type,
                    'json_value': json_value,
                })

        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['key', 'data_type', 'json_value'])
            writer.writeheader()
            writer.writerows(rows)

    def get_cached_value(self, key, data_type: CacheType):
        if key not in self.cache:
            return None
        return self.cache[key].get(data_type.value)

    def add_cached_value(self, key, data_type: CacheType, value, save_interval=1):
        if key not in self.cache:
            self.cache[key] = {}

        self.cache[key][data_type.value] = value

        self.changes += 1
        if self.changes >= save_interval:
            self._save_to_disk()
            self.changes = 0
