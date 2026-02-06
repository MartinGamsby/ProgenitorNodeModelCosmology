import csv
import json
import os
import pickle
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
    PICKLE = "pkl"

class Cache:
    _LOADERS = {
        CacheFormat.JSON: '_load_json',
        CacheFormat.CSV: '_load_csv',
        CacheFormat.PICKLE: '_load_pickle',
    }
    _SAVERS = {
        CacheFormat.JSON: '_save_json',
        CacheFormat.CSV: '_save_csv',
        CacheFormat.PICKLE: '_save_pickle',
    }

    def __init__(self, name="cache", format: CacheFormat = CacheFormat.CSV, _data_dir="data"):
        self.format = format
        self.name = name
        self._data_dir = _data_dir
        self.filepath = os.path.join(_data_dir, name + "." + format.value)
        self.changes = 0
        folder_path = os.path.dirname(self.filepath)

        # Only try to create if a folder path actually exists (handling strictly filenames)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)
        # ------------------------------------

        self.cache = self._load_from_disk()

    def _filepath_for(self, fmt):
        return os.path.join(self._data_dir, self.name + "." + fmt.value)

    def _load_with(self, fmt, filepath):
        return getattr(self, self._LOADERS[fmt])(filepath)

    def _load_from_disk(self):
        # Try primary format
        if os.path.exists(self.filepath):
            try:
                return self._load_with(self.format, self.filepath)
            except Exception:
                return {}

        # Fallback: try other formats
        for fmt in CacheFormat:
            if fmt == self.format:
                continue
            alt_path = self._filepath_for(fmt)
            if os.path.exists(alt_path):
                try:
                    return self._load_with(fmt, alt_path)
                except Exception:
                    continue

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
                    entry = {}
                    for col, val in row.items():
                        if col == 'key' or val == '':
                            continue
                        parsed = json.loads(val)
                        if '.' in col:
                            data_type, field = col.split('.', 1)
                            if data_type not in entry:
                                entry[data_type] = {}
                            entry[data_type][field] = parsed
                        else:
                            entry[col] = parsed
                    if entry:
                        cache[key] = entry
            except (KeyError, json.JSONDecodeError):
                return {}
        return cache

    def _load_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _save_to_disk(self):
        getattr(self, self._SAVERS[self.format])()

    def _save_json(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.cache, f, indent=4, cls=EnhancedJSONEncoder)

    def _flatten_value(self, data_type, value):
        """Flatten a value into column-name → string pairs.

        Scalars: {"velocity": "1.14"}
        Dicts:   {"metrics.match_avg_pct": "95.5", "metrics.match_end_pct": "98.2"}
        Dataclasses are converted to dicts first by EnhancedJSONEncoder.
        """
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        if isinstance(value, dict):
            return {f"{data_type}.{k}": json.dumps(v, cls=EnhancedJSONEncoder) for k, v in value.items()}
        return {data_type: json.dumps(value, cls=EnhancedJSONEncoder)}

    def _save_csv(self):
        # Build rows and collect all column names
        all_columns = set()
        flat_rows = []
        for key, data_types in self.cache.items():
            flat = {'key': key}
            for data_type, value in data_types.items():
                flat.update(self._flatten_value(data_type, value))
            flat_rows.append(flat)
            all_columns.update(flat.keys())

        all_columns.discard('key')
        fieldnames = ['key'] + sorted(all_columns)

        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(flat_rows)

    def _save_pickle(self):
        # Convert dataclasses to dicts before pickling for consistency
        data = {}
        for key, data_types in self.cache.items():
            data[key] = {}
            for dt, value in data_types.items():
                if dataclasses.is_dataclass(value):
                    data[key][dt] = dataclasses.asdict(value)
                else:
                    data[key][dt] = value
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)

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
