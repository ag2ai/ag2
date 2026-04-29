# Deserialization Security

AG2 uses JSON as the default serialization format for all network and cache
payloads.  Pickle is disabled by default because `pickle.loads` on
attacker-controlled bytes allows arbitrary code execution (RCE).

## Affected components

| Component | File | Env var to opt in |
|-----------|------|-------------------|
| Redis pub/sub serializer | `autogen/beta/streams/redis/serializer.py` | `AG2_ALLOW_PICKLE_DESERIALIZATION=1` |
| Redis cache | `autogen/cache/redis_cache.py` | `AG2_ALLOW_PICKLE_CACHE_READ=1` |
| Cosmos DB cache | `autogen/cache/cosmos_db_cache.py` | `AG2_ALLOW_PICKLE_CACHE_READ=1` |
| Teachability memo store | `autogen/agentchat/contrib/capabilities/teachability.py` | requires migration |

## Redis pub/sub serializer

The `Serializer.PICKLE` enum value is preserved for API compatibility but the
pickle path now raises `ValueError` unless `AG2_ALLOW_PICKLE_DESERIALIZATION=1`
is set in the environment.

Switch to JSON (the default) if you do not have an explicit need for pickle:

```python
from autogen.beta.streams.redis.serializer import Serializer

# Before (unsafe)
stream = RedisStream(..., serializer=Serializer.PICKLE)

# After (safe default)
stream = RedisStream(..., serializer=Serializer.JSON)
```

## Cache stores (Redis and Cosmos DB)

New writes always use a JSON payload with a one-byte version prefix.
Existing pickle payloads can be read back only with the opt-in env var:

```bash
export AG2_ALLOW_PICKLE_CACHE_READ=1   # temporary -- for migration only
```

To migrate existing cache data, re-populate the cache (call `cache.set` again)
while the env var is set.  Once migrated, remove the env var.

See `docs/cache-migration.md` for step-by-step instructions.

## Teachability memo store

The legacy `uid_text_dict.pkl` format is no longer loaded.  If a `.pkl` file
exists without a corresponding `.json` file, `MemoStore.__init__` raises
`RuntimeError` with instructions to run the migration helper:

```bash
python -m autogen.agentchat.contrib.capabilities.teachability_migrate_pickle_to_json \
    --path /path/to/db_dir
```

## Class resolver (JSON deserialization)

The JSON deserializer previously used `importlib.import_module` on the
`__type__` field from untrusted JSON payloads.  This allowed an attacker
controlling the Redis channel to trigger import of arbitrary modules.

The resolver now uses a compile-time registry.  All `BaseEvent` subclasses
are auto-registered at import time.  Custom event types must be registered
before they can be deserialized:

```python
from autogen.beta.streams.redis.serializer import register_event_class

@register_event_class
class MyCustomEvent(BaseEvent):
    ...
```
