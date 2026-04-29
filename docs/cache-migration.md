# Cache Migration -- Pickle to JSON

AG2's cache stores (RedisCache and CosmosDBCache) now serialize values as
JSON instead of pickle.  This document describes how to migrate existing
pickle-serialized caches.

## Why

`pickle.loads` on data retrieved from Redis or Cosmos DB allows remote code
execution for any actor with write access to the backend.  JSON serialization
eliminates this attack surface.

## Format versioning

Each payload is prefixed with a single byte:

| Prefix byte | Format |
|-------------|--------|
| `\x01`      | JSON (current, default) |
| `\x00`      | Legacy pickle (read-only, requires opt-in) |

New writes always produce `\x01` + JSON bytes, regardless of environment
variables.

## Migration steps

### Option A -- Flush and repopulate (recommended)

If the cache is ephemeral (used for LLM response caching), simply clear it:

```bash
# Redis
redis-cli FLUSHDB

# Cosmos DB -- delete and recreate the container via Azure portal or CLI
```

On next use, AG2 will write JSON-encoded entries automatically.

### Option B -- Read-back existing data and re-write

1. Set `AG2_ALLOW_PICKLE_CACHE_READ=1` in the environment.
2. Read each key and write it back:

```python
import os
os.environ["AG2_ALLOW_PICKLE_CACHE_READ"] = "1"

from autogen.cache.redis_cache import RedisCache

with RedisCache(seed="your_seed", redis_url="redis://localhost:6379/0") as cache:
    # Iterate your known keys and re-write them
    for key in your_key_list:
        value = cache.get(key)
        if value is not None:
            cache.set(key, value)  # re-writes as JSON
```

3. Remove `AG2_ALLOW_PICKLE_CACHE_READ` from the environment.

### Teachability memo store

The teachability store uses a dedicated migration helper:

```bash
python -m autogen.agentchat.contrib.capabilities.teachability_migrate_pickle_to_json \
    --path /path/to/db_dir
```

This reads `uid_text_dict.pkl`, writes `uid_text_dict.json`, and renames the
old file to `uid_text_dict.pkl.bak`.  The `.bak` file can be deleted once you
have confirmed the migration succeeded.

## Rollback

If you need to roll back to a version of AG2 that used pickle, restore the
`.pkl` file from backup.  The new version will refuse to load it unless
`AG2_ALLOW_PICKLE_CACHE_READ=1` is set.
