from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from app.core.schemas import Usage
from app.observability.logging import get_logger
from app.providers.base import Deployment, Pricing

logger = get_logger(__name__)


class PriceTable:
    """Loads public-model prices and optional provider-specific overrides."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.models: dict[str, Pricing] = {}
        self.provider_overrides: dict[str, dict[str, Pricing]] = {}
        self.path: Path | None = None
        if path is not None:
            self.load(path)

    def load(self, path: str | Path) -> None:
        source = Path(path)
        with source.open(encoding="utf-8") as file:
            document = yaml.safe_load(file) or {}
        if not isinstance(document, Mapping):
            raise ValueError("Pricing file must contain a model mapping")
        model_data = document.get("models", document)
        if not isinstance(model_data, Mapping):
            raise ValueError("Pricing file 'models' must be a mapping")

        models: dict[str, Pricing] = {}
        overrides: dict[str, dict[str, Pricing]] = {}
        for model, raw_config in model_data.items():
            if not isinstance(model, str) or not isinstance(raw_config, Mapping):
                continue
            config = dict(raw_config)
            providers = config.pop("providers", config.pop("provider_overrides", {}))
            base_fields = {
                key: config[key]
                for key in ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok")
                if key in config
            }
            if base_fields:
                models[model] = Pricing.model_validate(base_fields)
            if isinstance(providers, Mapping):
                model_overrides: dict[str, Pricing] = {}
                for provider, raw_override in providers.items():
                    if isinstance(provider, str) and isinstance(raw_override, Mapping):
                        model_overrides[provider] = Pricing.model_validate(dict(raw_override))
                if model_overrides:
                    overrides[model] = model_overrides
        self.models = models
        self.provider_overrides = overrides
        self.path = source

    def get(self, model: str, provider: str | None = None) -> Pricing | None:
        if provider is not None:
            override = self.provider_overrides.get(model, {}).get(provider)
            if override is not None:
                return override
        return self.models.get(model)

    def estimate_cost(
        self,
        model: str,
        provider: str,
        usage: Usage | Mapping[str, int],
        deployment: Deployment | None = None,
    ) -> float:
        pricing = self.get(model, provider)
        if pricing is None and deployment is not None:
            pricing = deployment.pricing
        if pricing is None:
            logger.warning("No pricing configured for model", model=model, provider=provider)
            return 0.0
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        cached_tokens = _usage_value(usage, "cached_tokens")
        return pricing.estimate(prompt_tokens, completion_tokens, cached_tokens)

    def estimate_savings(
        self,
        model: str,
        provider: str,
        usage: Usage | Mapping[str, int],
        deployment: Deployment | None = None,
    ) -> float:
        pricing = self.get(model, provider)
        if pricing is None and deployment is not None:
            pricing = deployment.pricing
        if pricing is None or pricing.cached_input_per_mtok is None:
            return 0.0
        cached_tokens = min(
            _usage_value(usage, "cached_tokens"), _usage_value(usage, "prompt_tokens")
        )
        rate_difference = max(pricing.input_per_mtok - pricing.cached_input_per_mtok, 0.0)
        return rate_difference * cached_tokens / 1_000_000


def _usage_value(usage: Usage | Mapping[str, int], name: str) -> int:
    if isinstance(usage, Mapping):
        return max(int(usage.get(name, 0)), 0)
    return max(int(getattr(usage, name)), 0)


_default_price_table: PriceTable | None = None


def _default_table() -> PriceTable:
    global _default_price_table
    if _default_price_table is None:
        _default_price_table = PriceTable(Path("config/pricing.yaml"))
    return _default_price_table


def estimate_cost(
    model: str,
    provider: str,
    usage: Usage | Mapping[str, int],
    deployment: Deployment | None = None,
    price_table: PriceTable | None = None,
) -> float:
    return (price_table or _default_table()).estimate_cost(model, provider, usage, deployment)


def estimate_savings(
    model: str,
    provider: str,
    usage: Usage | Mapping[str, int],
    deployment: Deployment | None = None,
    price_table: PriceTable | None = None,
) -> float:
    return (price_table or _default_table()).estimate_savings(model, provider, usage, deployment)
