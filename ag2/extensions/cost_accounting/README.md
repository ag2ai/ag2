# Cost Accounting Extension

Estimate model usage cost from AG2 `Usage` data and a maintained pricing catalog.

AG2 core usage accounting remains factual: providers report token usage, AG2
normalizes it into `Usage`, and this extension derives estimated cost from a
catalog. The estimate is not an invoice and does not replace provider billing
systems.

## LiteLLM-style catalog

The catalog loader accepts the model entries used by LiteLLM's
`model_prices_and_context_window.json`:

```python
from ag2.extensions.cost_accounting import CostCatalog, UsageCostEstimator

catalog = CostCatalog.from_litellm_mapping({
    "openai/gpt-4.1-nano": {
        "litellm_provider": "openai",
        "input_cost_per_token": 0.0000001,
        "cache_read_input_token_cost": 0.000000025,
        "cache_creation_input_token_cost": 0.000000125,
        "output_cost_per_token": 0.0000004,
    }
})

estimator = UsageCostEstimator(catalog)
estimate = estimator.estimate_usage(usage, model="gpt-4.1-nano", provider="openai")
```

## Observer

Use `CostAccountingObserver` to emit `ObserverAlert` events when estimated cost
crosses a threshold. The observer watches `UsageEvent`, AG2's source of truth
for token accounting, so sub-agent rollups are not double counted.

```python
from ag2.extensions.cost_accounting import CostAccountingObserver

observer = CostAccountingObserver(
    estimator,
    warn_threshold_usd="1.00",
    alert_threshold_usd="5.00",
)
```

