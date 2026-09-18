import hashlib
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from matdiscovery import immutable_hash_cache as module


def test_default_does_not_cache_or_touch_paths(tmp_path):
    assert module.maybe_cached_file_sha256(tmp_path / "absent") is None


def test_identical_file_reuse_and_nested_context_reset(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"fixed artifact")
    time.sleep(module.RECENT_WRITE_GUARD_NS / 1e9 + .05)
    with module.immutable_hash_cache() as counters:
        first = module.maybe_cached_file_sha256(path)
        with module.immutable_hash_cache() as inner:
            assert inner is counters
            assert module.maybe_cached_file_sha256(path) == first
        assert first == hashlib.sha256(path.read_bytes()).hexdigest()
        assert counters == {"files_hashed": 1, "bytes_read": 14, "cache_hits": 1}
    assert module.maybe_cached_file_sha256(path) is None


def test_recent_file_is_rechecked_and_never_cached(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"fixed")
    with module.immutable_hash_cache() as counters:
        first = module.maybe_cached_file_sha256(path)
        assert module.maybe_cached_file_sha256(path) == first
        assert counters["cache_hits"] == 0
        assert counters["files_hashed"] == 2
        assert counters["bytes_read"] == 4 * len(b"fixed")


def test_missing_or_future_metadata_never_qualifies_for_reuse():
    now = time.time_ns()
    assert not module.metadata_allows_hash_reuse(0, now)
    assert not module.metadata_allows_hash_reuse(now, now + 10_000_000_000)


def test_same_size_edit_with_restored_mtime_is_rehashed(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"before")
    with module.immutable_hash_cache() as counters:
        old = path.stat()
        first = module.maybe_cached_file_sha256(path)
        path.write_bytes(b"after!")
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        assert module.maybe_cached_file_sha256(path) != first
        assert counters["files_hashed"] == 2


def test_atomic_replacement_invalidates_same_path_and_mtime(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"before")
    with module.immutable_hash_cache() as counters:
        old = path.stat()
        first = module.maybe_cached_file_sha256(path)
        replacement = tmp_path / "new"
        replacement.write_bytes(b"after!")
        os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
        replacement.replace(path)
        assert module.maybe_cached_file_sha256(path) != first
        assert counters["files_hashed"] == 2


def test_symlink_retarget_resolves_new_content(tmp_path):
    a, b, link = [tmp_path / x for x in ("a", "b", "link")]
    a.write_bytes(b"first")
    b.write_bytes(b"other")
    link.symlink_to(a)
    with module.immutable_hash_cache() as counters:
        first = module.maybe_cached_file_sha256(link)
        link.unlink()
        link.symlink_to(b)
        assert module.maybe_cached_file_sha256(link) != first
        assert counters["files_hashed"] == 2


def test_mutation_during_hash_refuses_stale_digest(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"before")
    original = module._stream_sha256
    def mutate(handle, counters):
        digest = original(handle, counters)
        Path(handle.name).write_bytes(b"after!")
        return digest
    with module.immutable_hash_cache() as counters:
        with patch.object(module, "_stream_sha256", side_effect=mutate):
            with pytest.raises(ValueError, match="changed during hashing"):
                module.maybe_cached_file_sha256(path)
        assert counters["files_hashed"] == 0
        assert module.maybe_cached_file_sha256(path) == hashlib.sha256(b"after!").hexdigest()


def test_forked_process_does_not_inherit_cached_digest(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"fixed")
    with module.immutable_hash_cache():
        module.maybe_cached_file_sha256(path)
        with patch.object(module.os, "getpid", return_value=os.getpid() + 1):
            assert module.maybe_cached_file_sha256(path) is None


def test_context_clears_after_exception(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"fixed")
    with pytest.raises(RuntimeError):
        with module.immutable_hash_cache():
            module.maybe_cached_file_sha256(path)
            raise RuntimeError("test interruption")
    assert module.maybe_cached_file_sha256(path) is None
