import os
import tempfile

# Keep test runs out of the user's real analysis cache.
os.environ["HITSYNC_CACHE"] = tempfile.mkdtemp(prefix="hitsync_cache_")
