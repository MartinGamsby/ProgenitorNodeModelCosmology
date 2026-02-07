import csv
import json
import os
import pickle
import re
import time
from enum import Enum
import dataclasses
import datetime as dt

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


def _pid_alive(pid):
    """Check whether a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class CacheLock:
    """File-based lock with PID staleness detection.

    Creates <filepath>.lock containing the owning PID.
    If an existing lock belongs to a dead process it is automatically broken.
    """
    def __init__(self, filepath):
        self.lockpath = filepath + ".lock"
        self._owned = False

    def acquire(self):
        """Try to acquire the lock. Returns True if acquired, False if held by a live process."""
        try:
            fd = os.open(self.lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._owned = True
            return True
        except FileExistsError:
            try:
                with open(self.lockpath, 'r') as f:
                    owner_pid = int(f.read().strip())
                if not _pid_alive(owner_pid):
                    # Stale lock — break it and retry
                    try:
                        os.remove(self.lockpath)
                    except FileNotFoundError:
                        pass
                    return self.acquire()
                return False  # held by a live process
            except (ValueError, IOError):
                # Corrupt lock file — break it
                try:
                    os.remove(self.lockpath)
                except FileNotFoundError:
                    pass
                return self.acquire()

    def release(self):
        if not self._owned:
            return
        try:
            with open(self.lockpath, 'r') as f:
                owner_pid = int(f.read().strip())
            if owner_pid == os.getpid():
                os.remove(self.lockpath)
        except (FileNotFoundError, ValueError, IOError):
            pass
        self._owned = False

    @property
    def owner_pid(self):
        """Return the PID holding the lock, or None."""
        try:
            with open(self.lockpath, 'r') as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError, IOError):
            return None


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
        self.last_change = dt.datetime.now()
        self._lock = CacheLock(self.filepath)
        self.read_only = False
        folder_path = os.path.dirname(self.filepath)

        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        # Acquire lock for the lifetime of this Cache
        if not self._lock.acquire():
            owner = self._lock.owner_pid
            print(f"\n*** WARNING: Cache '{name}' is locked by PID {owner}. ***")
            print(f"    Lock file: {self._lock.lockpath}")
            try:
                answer = input("    Continue in read-only mode? [Y/n/kill] ").strip().lower()
            except (EOFError, OSError):
                answer = 'y'
            if answer == 'kill':
                print(f"    Killing PID {owner}...")
                try:
                    import signal
                    os.kill(owner, signal.SIGTERM)
                    time.sleep(0.5)
                except (OSError, ProcessLookupError):
                    pass
                if not self._lock.acquire():
                    print("    Still locked. Continuing in read-only mode.")
                    self.read_only = True
                else:
                    print("    Lock acquired.")
            elif answer in ('n', 'no'):
                raise RuntimeError(f"Cache '{name}' is locked by PID {owner}. Aborting.")
            else:
                self.read_only = True
                print("    Running in read-only mode. Changes will NOT be saved.\n")

        self.cache = self._load_from_disk()

    def close(self):
        """Save and release the lock. Called automatically by __del__."""
        if not self.read_only:
            try:
                self._save_to_disk()
            except Exception:
                pass
        self._lock.release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
                    key = self._join_key(row)
                    entry = {}
                    for col, val in row.items():
                        if col.startswith('key.') or val == '':
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
        if self.read_only:
            return
        getattr(self, self._SAVERS[self.format])()

    def _save_json(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.cache, f, indent=4, cls=EnhancedJSONEncoder)

    @staticmethod
    def _split_key(key):
        """Split a cache key into {column_name: value} for CSV columns.

        Each underscore-delimited part is split at the boundary after the
        last digit.  E.g. "200p" → ("200", "p"), "5.8-2.2Gyr" → ("5.8-2.2", "Gyr").
        Parts with no alpha suffix use their raw text as the column label.
        Column names are prefixed with position index for order preservation:
        key.0_p, key.1_Gyr, key.2_M, ...
        """
        parts = key.split('_')
        cols = {}
        for i, part in enumerate(parts):
            m = re.match(r'^(.*\d)(\D+)$', part)
            if m:
                value, suffix = m.group(1), m.group(2)
                cols[f"key.{i}_{suffix}"] = value
            else:
                cols[f"key.{i}_{part}"] = part
        return cols

    @staticmethod
    def _join_key(row):
        """Reconstruct the original cache key from key.* CSV columns."""
        indexed = []
        for col, val in row.items():
            if not col.startswith('key.'):
                continue
            rest = col[4:]  # strip "key."
            idx_str, _, suffix = rest.partition('_')
            idx = int(idx_str)
            if val == suffix:
                indexed.append((idx, val))
            else:
                indexed.append((idx, val + suffix))
        indexed.sort()
        return '_'.join(v for _, v in indexed)

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
        all_columns = set()
        flat_rows = []
        for key, data_types in self.cache.items():
            flat = self._split_key(key)
            for data_type, value in data_types.items():
                flat.update(self._flatten_value(data_type, value))
            flat_rows.append(flat)
            all_columns.update(flat.keys())

        key_cols = sorted(c for c in all_columns if c.startswith('key.'))
        data_cols = sorted(c for c in all_columns if not c.startswith('key.'))
        fieldnames = key_cols + data_cols

        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(flat_rows)

    def _save_pickle(self):
        data = {}
        for key, data_types in self.cache.items():
            data[key] = {}
            for data_type, value in data_types.items():
                if dataclasses.is_dataclass(value):
                    data[key][data_type] = dataclasses.asdict(value)
                else:
                    data[key][data_type] = value
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)

    def get_cached_value(self, key, data_type: CacheType):
        if key not in self.cache:
            return None
        return self.cache[key].get(data_type.value)

    def add_cached_value(self, key, data_type: CacheType, value, save_interval_s=5):
        if key not in self.cache:
            self.cache[key] = {}

        self.cache[key][data_type.value] = value

        if self.read_only:
            return

        if (dt.datetime.now() - self.last_change).total_seconds() >= save_interval_s:
            self._save_to_disk()
            self.last_change = dt.datetime.now()
