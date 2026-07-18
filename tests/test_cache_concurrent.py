"""Tests for the concurrent-safe Cache mode (shared cache for parallel sweeps).

Concurrent mode (Cache(concurrent=True), or env HMEA_CACHE_CONCURRENT=1) drops the
exclusive lifetime lock and instead does a read-MERGE-write under a short lock with an
atomic rename on each save, so several processes can SHARE one cache file without losing
entries and without the read-only/input() prompt. The default (exclusive) mode must stay
byte-identical -- single-worker runs and the rest of the suite never set the env var.
"""
import os

import pytest

from cosmo.cache import Cache, CacheLock, CacheType, CacheFormat


def test_default_is_exclusive_with_lifetime_lock(tmp_path):
    """Unset env / no flag => exclusive mode (lifetime lock held while open)."""
    c = Cache("excl", _data_dir=str(tmp_path))
    assert c.concurrent is False
    assert os.path.exists(c._lock.lockpath)  # lifetime lock held
    c.close()
    assert not os.path.exists(c._lock.lockpath)  # released on close


def test_env_var_enables_concurrent(tmp_path, monkeypatch):
    monkeypatch.setenv("HMEA_CACHE_CONCURRENT", "1")
    c = Cache("viaenv", _data_dir=str(tmp_path))
    assert c.concurrent is True
    # No lifetime lock is taken in concurrent mode.
    assert not os.path.exists(c._lock.lockpath)
    c.close()


def test_explicit_flag_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HMEA_CACHE_CONCURRENT", "1")
    c = Cache("explicit", _data_dir=str(tmp_path), concurrent=False)
    assert c.concurrent is False
    c.close()


def test_two_concurrent_caches_open_no_readonly(tmp_path):
    """The whole point: a 2nd concurrent Cache on the same file is NOT forced read-only."""
    a = Cache("shared", _data_dir=str(tmp_path), concurrent=True)
    b = Cache("shared", _data_dir=str(tmp_path), concurrent=True)
    assert not a.read_only
    assert not b.read_only
    a.close()
    b.close()


def test_concurrent_merge_no_lost_entries(tmp_path):
    """Two concurrent caches each add a distinct key; both survive after merge-on-close."""
    a = Cache("merge", _data_dir=str(tmp_path), concurrent=True)
    b = Cache("merge", _data_dir=str(tmp_path), concurrent=True)
    a.add_cached_value("k1", CacheType.METRICS, {"match_avg_pct": 95.0})
    b.add_cached_value("k2", CacheType.METRICS, {"match_avg_pct": 42.0})
    a.close()  # writes k1
    b.close()  # reads disk (k1), merges k2 -> k1 + k2

    fresh = Cache("merge", _data_dir=str(tmp_path), concurrent=True)
    assert fresh.get_cached_value("k1", CacheType.METRICS) == {"match_avg_pct": 95.0}
    assert fresh.get_cached_value("k2", CacheType.METRICS) == {"match_avg_pct": 42.0}
    fresh.close()


def test_concurrent_merge_picks_up_other_process_entries_into_memory(tmp_path):
    """After a save, a process's in-memory cache reflects the merged (union) view."""
    a = Cache("union", _data_dir=str(tmp_path), concurrent=True)
    a.add_cached_value("ka", CacheType.METRICS, {"v": 1})
    a.close()

    b = Cache("union", _data_dir=str(tmp_path), concurrent=True)
    b.add_cached_value("kb", CacheType.METRICS, {"v": 2})
    b.close()  # merge -> b.cache now holds both

    assert b.get_cached_value("ka", CacheType.METRICS) == {"v": 1}
    assert b.get_cached_value("kb", CacheType.METRICS) == {"v": 2}


def test_concurrent_save_leaves_no_tmp_files(tmp_path):
    c = Cache("notmp", _data_dir=str(tmp_path), concurrent=True)
    c.add_cached_value("k", CacheType.VELOCITY, 1.25)
    c.close()
    leftovers = [n for n in os.listdir(str(tmp_path)) if ".tmp" in n]
    assert leftovers == []
    # The real file exists and round-trips.
    fresh = Cache("notmp", _data_dir=str(tmp_path), concurrent=True)
    assert fresh.get_cached_value("k", CacheType.VELOCITY) == 1.25
    fresh.close()


def test_acquire_blocking_breaks_dead_pid_lock(tmp_path):
    lock = CacheLock(os.path.join(str(tmp_path), "deadpid"))
    with open(lock.lockpath, "w") as f:
        f.write("999999")  # a PID that is not alive
    assert lock.acquire_blocking(timeout=3.0) is True
    lock.release()
    assert not os.path.exists(lock.lockpath)


def test_acquire_blocking_reclaims_own_stale_lock(tmp_path):
    lock = CacheLock(os.path.join(str(tmp_path), "ownpid"))
    with open(lock.lockpath, "w") as f:
        f.write(str(os.getpid()))  # our own pid, left by a failed release
    assert lock.acquire_blocking(timeout=3.0) is True
    lock.release()


def test_acquire_blocking_times_out_when_held_by_live_other(tmp_path, monkeypatch):
    """A live, different-PID holder is waited out; acquire_blocking returns False on timeout."""
    lock = CacheLock(os.path.join(str(tmp_path), "live"))
    # Simulate a different, live owner: lock held by a pid that _pid_alive() reports alive.
    other = lock  # second handle on the same path
    with open(lock.lockpath, "w") as f:
        f.write("424242")
    monkeypatch.setattr("cosmo.cache._pid_alive", lambda pid: pid == 424242)
    assert other.acquire_blocking(timeout=1.0) is False
    # cleanup
    os.remove(lock.lockpath)


def test_heterogeneous_keys_in_one_file_roundtrip(tmp_path):
    """Keys of DIFFERENT structure sharing one file must each survive (no phantom-key
    corruption from empty key columns). This is the cube26-vs-virialized scenario."""
    c = Cache("hetero", _data_dir=str(tmp_path), concurrent=True)
    # Realistic structured keys with DIFFERENT shapes (extra slug on the 2nd).
    c.add_cached_value("100p_25M_5Gpc", CacheType.METRICS, {"chi2": 0.5})
    c.add_cached_value("100p_25M_5Gpc_virialized", CacheType.METRICS, {"chi2": 0.9})
    c.close()

    fresh = Cache("hetero", _data_dir=str(tmp_path), concurrent=True)
    assert fresh.get_cached_value("100p_25M_5Gpc", CacheType.METRICS) == {"chi2": 0.5}
    assert fresh.get_cached_value("100p_25M_5Gpc_virialized", CacheType.METRICS) == {"chi2": 0.9}
    fresh.close()


def test_exclusive_heterogeneous_accumulation_roundtrip(tmp_path):
    """The same heterogeneous-key fix also helps exclusive mode, where sequential
    'arm' runs accumulate differently-shaped keys into one shared metrics file."""
    a = Cache("acc", _data_dir=str(tmp_path))
    a.add_cached_value("400p_25M", CacheType.METRICS, {"chi2": 0.5})
    a.close()  # releases the lifetime lock
    b = Cache("acc", _data_dir=str(tmp_path))  # loads a's file, adds a different-shaped key
    b.add_cached_value("400p_25M_cube26", CacheType.METRICS, {"chi2": 0.6})
    b.close()

    fresh = Cache("acc", _data_dir=str(tmp_path))
    assert fresh.get_cached_value("400p_25M", CacheType.METRICS) == {"chi2": 0.5}
    assert fresh.get_cached_value("400p_25M_cube26", CacheType.METRICS) == {"chi2": 0.6}
    fresh.close()


def test_write_atomic_retries_os_replace(tmp_path, monkeypatch):
    """Windows: os.replace raises PermissionError if a concurrent reader has the
    destination open. _write_atomic must RETRY rather than crash the worker."""
    import os as _os
    c = Cache("retry", _data_dir=str(tmp_path), concurrent=True)
    c.add_cached_value("k", CacheType.VELOCITY, 1.0)
    real = _os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(_os, "replace", flaky)
    c._write_atomic()                 # must retry and eventually succeed
    monkeypatch.setattr(_os, "replace", real)
    assert calls["n"] >= 3
    c.close()
    fresh = Cache("retry", _data_dir=str(tmp_path), concurrent=True)
    assert fresh.get_cached_value("k", CacheType.VELOCITY) == 1.0
    fresh.close()


def test_merge_save_never_raises(tmp_path, monkeypatch):
    """A cache-save failure must be best-effort, never propagate (it would crash a
    sweep worker — the bug that killed 2 arms of the parallel run)."""
    import os as _os
    c = Cache("noraise", _data_dir=str(tmp_path), concurrent=True)
    c.add_cached_value("k", CacheType.VELOCITY, 1.0)
    monkeypatch.setattr(_os, "replace",
                        lambda s, d: (_ for _ in ()).throw(PermissionError(5, "denied")))
    c._merge_save()     # must NOT raise even though os.replace always fails
    c.close()           # must NOT raise
    # no leftover temp files
    assert [n for n in os.listdir(str(tmp_path)) if ".tmp" in n] == []


def test_concurrent_mode_csv_roundtrip_matches_exclusive(tmp_path):
    """Concurrent and exclusive modes produce the same on-disk content for the same data."""
    ex = Cache("rt_excl", _data_dir=str(tmp_path))
    ex.add_cached_value("kk", CacheType.METRICS, {"a": 1, "b": 2})
    ex.close()
    with open(ex.filepath) as f:
        excl_text = f.read()

    co = Cache("rt_conc", _data_dir=str(tmp_path), concurrent=True)
    co.add_cached_value("kk", CacheType.METRICS, {"a": 1, "b": 2})
    co.close()
    with open(co.filepath) as f:
        conc_text = f.read()

    assert excl_text == conc_text
