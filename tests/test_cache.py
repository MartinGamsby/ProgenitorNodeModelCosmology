"""Unit tests for cache module with JSON and CSV format support."""
import unittest
import os
import tempfile
import shutil
import json
from dataclasses import dataclass

from cosmo.cache import Cache, CacheType, CacheFormat, EnhancedJSONEncoder


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
        # Reload from disk to get the serialized dict form
        cache2 = self._make_cache()
        retrieved = cache2.get_cached_value("k1", CacheType.RESULTS)
        self.assertEqual(retrieved['size_final_Gpc'], 1.5)

    def test_persists_across_instances(self):
        c1 = self._make_cache()
        c1.add_cached_value("k1", CacheType.VELOCITY, 2.5)
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 2.5)

    def test_returns_none_for_missing_key(self):
        cache = self._make_cache()
        self.assertIsNone(cache.get_cached_value("nope", CacheType.VELOCITY))

    def test_returns_none_for_missing_data_type(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        self.assertIsNone(cache.get_cached_value("k1", CacheType.METRICS))

    def test_batched_save(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.0, save_interval=3)
        cache.add_cached_value("k2", CacheType.VELOCITY, 2.0, save_interval=3)
        # Not saved yet (2 < 3)
        c2 = self._make_cache()
        self.assertIsNone(c2.get_cached_value("k1", CacheType.VELOCITY))
        # Third add triggers save
        cache.add_cached_value("k3", CacheType.VELOCITY, 3.0, save_interval=3)
        c3 = self._make_cache()
        self.assertAlmostEqual(c3.get_cached_value("k1", CacheType.VELOCITY), 1.0)

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
        # Reload from disk to get the serialized dict form
        cache2 = self._make_cache()
        retrieved = cache2.get_cached_value("k1", CacheType.RESULTS)
        self.assertEqual(retrieved['size_final_Gpc'], 1.5)

    def test_persists_across_instances(self):
        c1 = self._make_cache()
        c1.add_cached_value("k1", CacheType.VELOCITY, 3.14)
        c2 = self._make_cache()
        self.assertAlmostEqual(c2.get_cached_value("k1", CacheType.VELOCITY), 3.14)

    def test_csv_header(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        with open(os.path.join(self.tmpdir, "test.csv"), 'r') as f:
            header = f.readline().strip()
        self.assertEqual(header, "key,data_type,json_value")

    def test_csv_multiple_data_types_same_key(self):
        cache = self._make_cache()
        cache.add_cached_value("k1", CacheType.VELOCITY, 1.5)
        cache.add_cached_value("k1", CacheType.METRICS, {'a': 1})
        self.assertAlmostEqual(cache.get_cached_value("k1", CacheType.VELOCITY), 1.5)
        self.assertEqual(cache.get_cached_value("k1", CacheType.METRICS), {'a': 1})

    def test_complex_nested_dict(self):
        cache = self._make_cache()
        data = {'level1': {'level2': [1, 2, 3]}, 'values': [1.5, 2.5]}
        cache.add_cached_value("k1", CacheType.METRICS, data)
        c2 = self._make_cache()
        retrieved = c2.get_cached_value("k1", CacheType.METRICS)
        self.assertEqual(retrieved['level1']['level2'], [1, 2, 3])
        self.assertEqual(retrieved['values'], [1.5, 2.5])


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
        # Create CSV with value 2.0
        c = Cache("fb3", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c.add_cached_value("k1", CacheType.VELOCITY, 2.0)
        # JSON request reads JSON
        j2 = Cache("fb3", format=CacheFormat.JSON, _data_dir=self.tmpdir)
        self.assertAlmostEqual(j2.get_cached_value("k1", CacheType.VELOCITY), 1.0)
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
        j.add_cached_value("k1", CacheType.VELOCITY, 1.0)
        # Load as CSV (fallback from JSON), then add a new value (triggers save as CSV)
        c = Cache("convert", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        c.add_cached_value("k2", CacheType.VELOCITY, 2.0)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "convert.csv")))

    def test_key_with_special_characters(self):
        """Cache keys with commas and quotes should survive CSV round-trip."""
        cache = Cache("special", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        weird_key = 'calibration_200p_5.8-13.8Gyr_1centerM_250steps_123seed_0.0rnd.'
        cache.add_cached_value(weird_key, CacheType.VELOCITY, 1.14)
        c2 = Cache("special", format=CacheFormat.CSV, _data_dir=self.tmpdir)
        self.assertAlmostEqual(c2.get_cached_value(weird_key, CacheType.VELOCITY), 1.14)


if __name__ == '__main__':
    unittest.main()
