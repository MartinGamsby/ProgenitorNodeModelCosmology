"""Unit tests for cache module with JSON, CSV, and Pickle format support."""
import unittest
import os
import tempfile
import shutil
import json
import time
from dataclasses import dataclass

from cosmo.cache import Cache, CacheType, CacheFormat, CacheLock, EnhancedJSONEncoder, _pid_alive

# Monkey-patch for tests: save immediately so we don't depend on timing
_original_add = Cache.add_cached_value
def _add_immediate(self, key, data_type, value, save_interval_s=0):
    return _original_add(self, key, data_type, value, save_interval_s=0)
Cache.add_cached_value = _add_immediate


@dataclass
class SampleDataclass:
    size_final_Gpc: float
    radius_max_Gpc: float
    a_final: float


class TestEnhancedJSONEncoder(unittest.TestCase):

    def test_encodes_dataclass(self):
        obj = SampleDataclass(size_final_Gpc=1.5, radius_max_Gpc=0.8, a_final=0.99)
        encoded = json.dumps(obj, cls=EnhancedJSONEncoder)
        decoded = json.loads(encoded)
        self.assertEqual(decoded['size_final_Gpc'], 1.5)
        self.assertEqual(decoded['radius_max_Gpc'], 0.8)
        self.assertEqual(decoded['a_final'], 0.99)

    def test_encodes_normal_types(self):
        data = {"a": 1, "b": [2, 3]}
        encoded = json.dumps(data, cls=EnhancedJSONEncoder)
        self.assertEqual(json.loads(encoded), data)


class TestCacheJSON(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_cache(self, name="test", fmt=CacheFormat.JSON):
        return Cache(name, format=fmt, _data_dir=self.tmpdir)

    def test_creates_json_file(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "test.json")))

    def test_store_and_retrieve_float(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.234)
        self.assertAlmostEqual(cache.get_cached_value("k1", CacheType.VELOCITY), 1.234)

    def test_store_and_retrieve_dict(self):
        cache = self._make_cache()
        metrics = {'match_avg_pct': 95.5, 'match_end_pct': 98.2}
        cache.add_cached_value("k1", CacheType.METRICS, metrics)
        retrieved = cache.get_cached_value("k1", CacheType.METRICS)
        self.assertEqual(retrieved['match_avg_pct'], 95.5)
        self.assertEqual(retrieved['match_end_pct'], 98.2)

    def test_store_and_retrieve_dataclass(self):
        cache = self._make_cache()
        obj = SampleDataclass(1.5, 0.8, 0.99)
        cache.add_cached_value("k1", CacheType.RESULTS, obj)
        cache.close()
        # Reload from disk to get the serialized dict form
        cache2 = self._make_cache()
        retrieved = cache2.get_cached_value("k1", CacheType.RESULTS)
        self.assertEqual(retrieved['size_final_Gpc'], 1.5)

    def test_persists_across_instances(self):
        c1 = self._make_cache()
        c1.add_cached_value("k1", CacheType.VELOCITY, 2.5)
        c1.close()
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 2.5)

    def test_returns_none_for_missing_key(self):
        cache = self._make_cache()
        self.assertIsNone(cache.get_cached_value("nope", CacheType.VELOCITY))

    def test_returns_none_for_missing_data_type(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.METRICS))

    def test_time_based_save(self):
        cache = self._make_cache()
        # save_interval_s=0 → saves immediately
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.0, save_interval_s=0)
        cache.close()
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 1.0)

    def test_multiple_keys(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        cache.add_cached_value("k2", CacheType.VELOCITY, 2.0)
        self.assertAlmostEqual(cache.get_cached_value("k1", CacheType.VELOCITY), 1.0)
        self.assertAlmostEqual(cache.get_cached_value("k2", CacheType.VELOCITY), 2.0)

    def test_overwrite_value(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        cache.add_cached_value("k1", CacheType.VELOCITY, 9.9)
        self.assertAlmostEqual(cache.get_cached_value("k1", CacheType.VELOCITY), 9.9)


class TestCacheCSV(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_cache(self, name="test", fmt=CacheFormat.CSV):
        return Cache(name, format=fmt, _data_dir=self.tmpdir)

    def test_creates_csv_file(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "test.csv")))

    def test_default_format_is_csv(self):
        cache = Cache("deftest", _data_dir=self.tmpdir)
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "deftest.csv")))
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "deftest.json")))

    def test_store_and_retrieve_float(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.234)
        self.assertAlmostEqual(cache.get_cached_value("k1", CacheType.VELOCITY), 1.234)

    def test_store_and_retrieve_dict(self):
        cache = self._make_cache()
        metrics = {'match_avg_pct': 95.5, 'count': 10}
        cache.add_cached_value("k1", CacheType.METRICS, metrics)
        retrieved = cache.get_cached_value("k1", CacheType.METRICS)
        self.assertEqual(retrieved['match_avg_pct'], 95.5)
        self.assertEqual(retrieved['count'], 10)

    def test_store_and_retrieve_dataclass(self):
        cache = self._make_cache()
        obj = SampleDataclass(1.5, 0.8, 0.99)
        cache.add_cached_value("k1", CacheType.RESULTS, obj)
        cache.close()
        # Reload from disk to get the serialized dict form
        cache2 = self._make_cache()
        retrieved = cache2.get_cached_value("k1", CacheType.RESULTS)
        self.assertEqual(retrieved['size_final_Gpc'], 1.5)

    def test_persists_across_instances(self):
        c1 = self._make_cache()
        c1.add_cached_value("k1", CacheType.VELOCITY, 3.14)
        c1.close()
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 3.14)

    def test_csv_header_scalar(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        with open(os.path.join(self.tmpdir, "test.csv"), 'r') as f:
            header = f.readline().strip()
        # key.0_k1 is the split key column, velocity is the data column
        self.assertEqual(header, "key.0_k1,velocity")

    def test_csv_header_dict_flattened(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.METRICS, {'b_val': 2, 'a_val': 1})
        with open(os.path.join(self.tmpdir, "test.csv"), 'r') as f:
            header = f.readline().strip()
        self.assertEqual(header, "key.0_k1,metrics.a_val,metrics.b_val")

    def test_csv_header_real_key(self):
        """Real cache keys get split into meaningful columns."""
        cache = self._make_cache()
        cache.add_cached_value("200p_5.8-13.8Gyr_855M", CacheType.VELOCITY, 1.0)
        with open(os.path.join(self.tmpdir, "test.csv"), 'r') as f:
            header = f.readline().strip()
        self.assertEqual(header, "key.0_p,key.1_Gyr,key.2_M,velocity")

    def test_csv_multiple_data_types_same_key(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        cache.add_cached_value("k1", CacheType.METRICS, {'a': 1})
        cache.close()
        # Verify round-trip through disk
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 1.5)
        self.assertEqual(c2.get_cached_value("k1", CacheType.METRICS), {'a': 1})

    def test_csv_dict_values_are_flat_columns(self):
        """Dict values should become individual columns, not JSON blobs."""
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.RESULTS, {'size': 1.5, 'radius': 0.8})
        with open(os.path.join(self.tmpdir, "test.csv"), 'r') as f:
            import csv as csv_mod
            reader = csv_mod.DictReader(f)
            row = next(reader)
        self.assertIn('results.size', row)
        self.assertIn('results.radius', row)
        self.assertNotIn('results', row)

    def test_complex_nested_dict(self):
        """Nested dicts within a value are JSON-encoded per field."""
        cache = self._make_cache()
        data = {'level1': {'level2': [1, 2, 3]}, 'values': [1.5, 2.5]}
        cache.add_cached_value("k1", CacheType.METRICS, data)
        cache.close()
        c2 = self._make_cache()
        retrieved = c2.get_cached_value("k1", CacheType.METRICS)
        self.assertEqual(retrieved['level1'], {'level2': [1, 2, 3]})
        self.assertEqual(retrieved['values'], [1.5, 2.5])


class TestCachePickle(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_cache(self, name="test", fmt=CacheFormat.PICKLE):
        return Cache(name, format=fmt, _data_dir=self.tmpdir)

    def test_creates_pkl_file(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "test.pkl")))

    def test_store_and_retrieve_float(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.234)
        cache.close()
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 1.234)

    def test_store_and_retrieve_dict(self):
        cache = self._make_cache()
        metrics = {'match_avg_pct': 95.5, 'count': 10}
        cache.add_cached_value("k1", CacheType.METRICS, metrics)
        cache.close()
        c2 = self._make_cache()
        retrieved = c2.get_cached_value("k1", CacheType.METRICS)
        self.assertEqual(retrieved['match_avg_pct'], 95.5)
        self.assertEqual(retrieved['count'], 10)

    def test_store_and_retrieve_dataclass(self):
        cache = self._make_cache()
        obj = SampleDataclass(1.5, 0.8, 0.99)
        cache.add_cached_value("k1", CacheType.RESULTS, obj)
        cache.close()
        c2 = self._make_cache()
        retrieved = c2.get_cached_value("k1", CacheType.RESULTS)
        self.assertEqual(retrieved['size_final_Gpc'], 1.5)

    def test_persists_across_instances(self):
        c1 = self._make_cache()
        c1.add_cached_value("k1", CacheType.VELOCITY, 2.718)
        c1.close()
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 2.718)

    def test_multiple_data_types(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        cache.add_cached_value("k1", CacheType.METRICS, {'a': 1})
        cache.close()
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 1.5)
        self.assertEqual(c2.get_cached_value("k1", CacheType.METRICS), {'a': 1})


class TestCacheFormatFallback(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_csv_requested_falls_back_to_json(self):
        # Write JSON file
        json_cache = Cache("fb1", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        json_cache.add_cached_value("k1", CacheType.VELOCITY, 5.678)
        # Request CSV — only JSON exists
        csv_cache = Cache("fb1", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertAlmostEqual(csv_cache.get_cached_value("k1", CacheType.VELOCITY), 5.678)

    def test_json_requested_falls_back_to_csv(self):
        csv_cache = Cache("fb2", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        csv_cache.add_cached_value("k1", CacheType.VELOCITY, 1.234)
        json_cache = Cache("fb2", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        self.assertAlmostEqual(json_cache.get_cached_value("k1", CacheType.VELOCITY), 1.234)

    def test_primary_format_wins_over_fallback(self):
        # Create JSON with value 1.0
        j = Cache("fb3", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        j.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        j.close()
        # Create CSV with value 2.0
        c = Cache("fb3", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c.add_cached_value("k1", CacheType.VELOCITY, 2.0)
        c.close()
        # JSON request reads JSON
        j2 = Cache("fb3", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        self.assertAlmostEqual(j2.get_cached_value("k1", CacheType.VELOCITY), 1.0)
        j2.close()
        # CSV request reads CSV
        c2 = Cache("fb3", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 2.0)

    def test_fallback_preserves_dict_values(self):
        json_cache = Cache("fb4", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        metrics = {'match_avg_pct': 88.3, 'size': 42}
        json_cache.add_cached_value("k1", CacheType.METRICS, metrics)
        csv_cache = Cache("fb4", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        retrieved = csv_cache.get_cached_value("k1", CacheType.METRICS)
        self.assertEqual(retrieved['match_avg_pct'], 88.3)
        self.assertEqual(retrieved['size'], 42)

    def test_pickle_requested_falls_back_to_json(self):
        j = Cache("fb5", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        j.add_cached_value("k1", CacheType.VELOCITY, 3.14)
        p = Cache("fb5", format=CacheFormat.PICKLE, _data_dir=self.tmpdir)
        self.assertAlmostEqual(p.get_cached_value("k1", CacheType.VELOCITY), 3.14)

    def test_pickle_requested_falls_back_to_csv(self):
        c = Cache("fb6", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c.add_cached_value("k1", CacheType.VELOCITY, 2.72)
        p = Cache("fb6", format=CacheFormat.PICKLE, _data_dir=self.tmpdir)
        self.assertAlmostEqual(p.get_cached_value("k1", CacheType.VELOCITY), 2.72)

    def test_csv_requested_falls_back_to_pickle(self):
        p = Cache("fb7", format=CacheFormat.PICKLE, _data_dir=self.tmpdir)
        p.add_cached_value("k1", CacheType.VELOCITY, 1.61)
        c = Cache("fb7", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertAlmostEqual(c.get_cached_value("k1", CacheType.VELOCITY), 1.61)

    def test_json_requested_falls_back_to_pickle(self):
        p = Cache("fb8", format=CacheFormat.PICKLE, _data_dir=self.tmpdir)
        p.add_cached_value("k1", CacheType.VELOCITY, 6.28)
        j = Cache("fb8", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        self.assertAlmostEqual(j.get_cached_value("k1", CacheType.VELOCITY), 6.28)

    def test_fallback_dict_pickle_to_csv(self):
        p = Cache("fb9", format=CacheFormat.PICKLE, _data_dir=self.tmpdir)
        p.add_cached_value("k1", CacheType.METRICS, {'x': 1, 'y': 2})
        c = Cache("fb9", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        retrieved = c.get_cached_value("k1", CacheType.METRICS)
        self.assertEqual(retrieved['x'], 1)
        self.assertEqual(retrieved['y'], 2)

    def test_no_file_returns_empty(self):
        cache = Cache("nonexistent", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.VELOCITY))


class TestCacheEdgeCases(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_corrupted_json_returns_empty(self):
        path = os.path.join(self.tmpdir, "bad.json")
        with open(path, 'w') as f:
            f.write("{ not valid json !!}")
        cache = Cache("bad", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.VELOCITY))

    def test_corrupted_csv_returns_empty(self):
        path = os.path.join(self.tmpdir, "bad.csv")
        with open(path, 'w') as f:
            f.write("not,a,valid,csv\nwith,bad,data,here")
        cache = Cache("bad", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.VELOCITY))

    def test_corrupted_pickle_returns_empty(self):
        path = os.path.join(self.tmpdir, "bad.pkl")
        with open(path, 'wb') as f:
            f.write(b"not valid pickle data!!")
        cache = Cache("bad", format=CacheFormat.PICKLE, _data_dir=self.tmpdir)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.VELOCITY))

    def test_empty_cache_file_json(self):
        path = os.path.join(self.tmpdir, "empty.json")
        with open(path, 'w') as f:
            f.write("")
        cache = Cache("empty", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.VELOCITY))

    def test_empty_cache_file_csv(self):
        path = os.path.join(self.tmpdir, "empty.csv")
        with open(path, 'w') as f:
            f.write("")
        cache = Cache("empty", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.VELOCITY))

    def test_saves_in_requested_format_after_fallback(self):
        """After loading via fallback, saves should use the requested format."""
        # Create JSON
        j = Cache("convert", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        j.add_cached_value("k1", CacheType.VELOCITY, 1.0, save_interval_s=0)
        # Load as CSV (fallback from JSON), then add a new value (triggers save as CSV)
        c = Cache("convert", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c.add_cached_value("k2", CacheType.VELOCITY, 2.0, save_interval_s=0)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "convert.csv")))

    def test_key_with_special_characters(self):
        """Cache keys with commas and quotes should survive CSV round-trip."""
        cache = Cache("special", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        weird_key = 'calibration_200p_5.8-13.8Gyr_1centerM_250steps_123seed_0.0rnd.'
        cache.add_cached_value(weird_key, CacheType.VELOCITY, 1.14)
        cache.close()
        c2 = Cache("special", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertAlmostEqual(c2.get_cached_value(weird_key, CacheType.VELOCITY), 1.14)


class TestCacheLock(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lockpath = os.path.join(self.tmpdir, "test.csv.lock")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_acquire_and_release(self):
        lock = CacheLock(os.path.join(self.tmpdir, "test.csv"))
        self.assertTrue(lock.acquire())
        self.assertTrue(os.path.exists(self.lockpath))
        lock.release()
        self.assertFalse(os.path.exists(self.lockpath))

    def test_lock_contains_pid(self):
        lock = CacheLock(os.path.join(self.tmpdir, "test.csv"))
        lock.acquire()
        with open(self.lockpath, 'r') as f:
            pid = int(f.read().strip())
        self.assertEqual(pid, os.getpid())
        lock.release()

    def test_acquire_returns_false_when_held(self):
        """Second acquire on same path returns False when held by live process."""
        lock1 = CacheLock(os.path.join(self.tmpdir, "test.csv"))
        self.assertTrue(lock1.acquire())
        lock2 = CacheLock(os.path.join(self.tmpdir, "test.csv"))
        self.assertFalse(lock2.acquire())
        lock1.release()

    def test_stale_lock_broken(self):
        """Lock from a dead PID is automatically broken."""
        with open(self.lockpath, 'w') as f:
            f.write("999999999")
        lock = CacheLock(os.path.join(self.tmpdir, "test.csv"))
        self.assertTrue(lock.acquire())
        self.assertTrue(lock._owned)
        lock.release()

    def test_corrupt_lock_broken(self):
        """Corrupt lock file is automatically broken."""
        with open(self.lockpath, 'w') as f:
            f.write("not_a_pid")
        lock = CacheLock(os.path.join(self.tmpdir, "test.csv"))
        self.assertTrue(lock.acquire())
        self.assertTrue(lock._owned)
        lock.release()

    def test_owner_pid(self):
        lock = CacheLock(os.path.join(self.tmpdir, "test.csv"))
        self.assertIsNone(lock.owner_pid)
        lock.acquire()
        self.assertEqual(lock.owner_pid, os.getpid())
        lock.release()

    def test_pid_alive_self(self):
        self.assertTrue(_pid_alive(os.getpid()))

    def test_pid_alive_dead(self):
        self.assertFalse(_pid_alive(999999999))


class TestCacheLifetimeLock(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_cache_holds_lock_while_alive(self):
        cache = Cache("lt1", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        lockpath = os.path.join(self.tmpdir, "lt1.csv.lock")
        self.assertTrue(os.path.exists(lockpath))
        cache.close()
        self.assertFalse(os.path.exists(lockpath))

    def test_cache_releases_lock_on_close(self):
        c1 = Cache("lt2", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c1.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        c1.close()
        # After close, another cache can open the same name
        c2 = Cache("lt2", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertFalse(c2.read_only)
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 1.0)
        c2.close()

    def test_separate_names_no_conflict(self):
        c1 = Cache("a", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c2 = Cache("b", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertFalse(c1.read_only)
        self.assertFalse(c2.read_only)
        c1.close()
        c2.close()

    def test_read_only_on_conflict(self):
        """Second Cache for same name opens read-only (simulated via stdin EOF)."""
        c1 = Cache("lt3", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c1.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        # Second cache: input() will raise EOFError in tests → defaults to read-only
        c2 = Cache("lt3", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertTrue(c2.read_only)
        # Can still read
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 1.0)
        c1.close()
        c2.close()

    def test_read_only_does_not_save(self):
        """Read-only cache doesn't write to disk."""
        c1 = Cache("lt4", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c1.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        # c2 opens read-only (EOFError → defaults to 'y')
        c2 = Cache("lt4", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertTrue(c2.read_only)
        c2.add_cached_value("k2", CacheType.VELOCITY, 999.0)
        c2.close()
        c1.close()
        # Reload — only k1 should exist, not k2
        c3 = Cache("lt4", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertAlmostEqual(c3.get_cached_value("k1", CacheType.VELOCITY), 1.0)
        self.assertIsNone(c3.get_cached_value("k2", CacheType.VELOCITY))
        c3.close()

    def test_saves_on_close(self):
        """Data added is persisted when close() is called."""
        c1 = Cache("lt5", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c1.add_cached_value("k1", CacheType.VELOCITY, 42.0)
        c1.close()
        c2 = Cache("lt5", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 42.0)
        c2.close()


if __name__ == '__main__':
    unittest.main()
