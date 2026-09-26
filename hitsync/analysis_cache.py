"""On-disk cache of media analysis, keyed by file identity.

Re-analysing a file that hasn't changed (same path, size and modification
time, same scan settings) loads the stored signals in a fraction of a second
instead of decoding it again. Entries live in the user's cache folder and are
plain JSON (the analysis ``to_dict`` form); a corrupt or stale entry is just
recomputed.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

# Bump when an analysis algorithm changes so old results aren't reused.
VERSION = {"audio": 3, "video": 2}
MAX_ENTRIES = 40


def cache_dir() -> str:
    base = os.environ.get("HITSYNC_CACHE")
    if not base:
        root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(root, "HitSync", "cache")
    return base


def fingerprint(kind: str, path: str, **settings) -> str:
    """Identity of a media file + the settings its analysis depends on.

    Empty when the file doesn't exist (never matches a cached result).
    """
    if not path or not os.path.isfile(path):
        return ""
    st = os.stat(path)
    ident = {"kind": kind, "v": VERSION[kind],
             "path": os.path.normcase(os.path.abspath(path)),
             "size": st.st_size, "mtime": st.st_mtime_ns, "settings": settings}
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()


def load(key: str) -> dict | None:
    if not key:
        return None
    try:
        with open(os.path.join(cache_dir(), key + ".json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save(key: str, data: dict):
    if not key:
        return
    d = cache_dir()
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, default=_json_default)
        os.replace(tmp, os.path.join(d, key + ".json"))
        _prune(d)
    except OSError:
        pass          # caching is best-effort


def _prune(d: str):
    files = [os.path.join(d, n) for n in os.listdir(d) if n.endswith(".json")]
    if len(files) <= MAX_ENTRIES:
        return
    files.sort(key=os.path.getmtime)
    for f in files[: len(files) - MAX_ENTRIES]:
        try:
            os.remove(f)
        except OSError:
            pass


def _json_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
