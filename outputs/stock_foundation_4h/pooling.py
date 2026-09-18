"""Shared token pooling used by the v2 FinBERT/ModernBERT comparison."""
from __future__ import annotations

import torch


def special_token_excluded_mean(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool real, non-padding, non-special tokens.

    The denominator is clamped only for an empty-content defensive fallback;
    the registered project inputs all contain at least one real token.
    """
    keep = attention_mask.bool() & ~special_tokens_mask.bool()
    weights = keep.unsqueeze(-1).to(hidden.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom
