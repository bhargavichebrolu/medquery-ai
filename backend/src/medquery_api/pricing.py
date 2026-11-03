"""Estimated cost of an LLM answer, from a dated price table (PRD 3.3).

Prices change, so every estimate carries the date its price was checked. A model without a
checked price gets no estimate rather than a guess; the mock provider costs nothing.
"""

from dataclasses import dataclass

from medquery_api.llm.base import TokenUsage

PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache writes
    checked_on: str  # when the price was compared with PRICING_SOURCE


# Checked against Anthropic's pricing page on 2 November 2026 (PRD 3.1). Standard API rates;
# batch, regional (inference_geo) and other variants are not used by this application.
PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(
        input=2.00, output=10.00, cache_read=0.10, cache_write=2.50, checked_on="2026-11-02"
    ),
}


@dataclass(frozen=True)
class CostEstimate:
    usd: float | None  # None when no checked price is known for the model
    checked_on: str | None
    note: str


def estimate_cost(provider: str, model: str, usage: TokenUsage) -> CostEstimate:
    if provider == "mock":
        return CostEstimate(usd=0.0, checked_on=None, note="The mock provider costs nothing.")
    price = PRICES.get(model)
    if price is None:
        return CostEstimate(
            usd=None,
            checked_on=None,
            note=f"No checked price for {model}, so no estimate is given.",
        )
    usd = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    ) / 1_000_000
    return CostEstimate(
        usd=usd,
        checked_on=price.checked_on,
        note=f"Estimate at standard API prices checked on {price.checked_on}; prices can change.",
    )
