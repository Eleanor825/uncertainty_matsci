"""Opt-in, process-local SHA reuse while immutable research inputs are audited.

Every call still resolves and stats its target. Cache entries require unchanged
device, inode, size, modification time AND change time. A mutation during a new
hash raises instead of publishing a stale digest. No structural or scientific
verification is skipped, and the default (outside this context) is uncached.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import stat
import time

_CURRENT = ContextVar("matdiscovery_immutable_hash_cache", default=None)
RECENT_WRITE_GUARD_NS = 2_000_000_000


def metadata_allows_hash_reuse(mtime_ns, ctime_ns):
    """Avoid same-tick metadata collisions for new/recently modified files."""
    if not isinstance(mtime_ns, int) or not isinstance(ctime_ns, int) or min(mtime_ns, ctime_ns) <= 0:
        return False
    return time.time_ns() - max(mtime_ns, ctime_ns) >= RECENT_WRITE_GUARD_NS


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _stream_sha256(handle, counters):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
        counters["bytes_read"] += len(block)
    return digest.hexdigest()


def maybe_cached_file_sha256(path):
    """Return None when disabled; otherwise verify identity and return SHA256."""
    current = _CURRENT.get()
    if current is None or current["pid"] != os.getpid():
        return None
    target = Path(path).resolve(strict=True)
    before = target.stat()
    if not stat.S_ISREG(before.st_mode):
        return None
    key = (str(target), *_identity(before))
    reusable = metadata_allows_hash_reuse(before.st_mtime_ns, before.st_ctime_ns)
    if reusable and key in current["digests"]:
        current["counters"]["cache_hits"] += 1
        return current["digests"][key]
    with target.open("rb") as handle:
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise ValueError("Research artifact changed before hashing: " + str(target))
        digest = _stream_sha256(handle, current["counters"])
        after_open = os.fstat(handle.fileno())
    if _identity(after_open) != _identity(before) or _identity(target.stat()) != _identity(before):
        raise ValueError("Research artifact changed during hashing: " + str(target))
    if not reusable:
        # Linux metadata can remain identical across writes within one clock
        # tick. Reopen and compare actual contents for these recent artifacts.
        with target.open("rb") as handle:
            corroborating = _stream_sha256(handle, current["counters"])
            second_stat = os.fstat(handle.fileno())
        if corroborating != digest or _identity(second_stat) != _identity(before) or _identity(target.stat()) != _identity(before):
            raise ValueError("Research artifact changed during hashing: " + str(target))
    else:
        current["digests"][key] = digest
    current["counters"]["files_hashed"] += 1
    return digest


@contextmanager
def immutable_hash_cache():
    """Reuse unchanged hashes within one explicit audit/run; expose I/O counters."""
    existing = _CURRENT.get()
    if existing is not None and existing["pid"] == os.getpid():
        yield existing["counters"]
        return
    counters = {"files_hashed": 0, "bytes_read": 0, "cache_hits": 0}
    token = _CURRENT.set({"pid": os.getpid(), "digests": {}, "counters": counters})
    try:
        yield counters
    finally:
        _CURRENT.reset(token)
