"""Verify the matched v2 encoder comparison and publish a narrow result."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "v2"
PRIVATE = ROOT / "work" / "stock-data" / "foundation_4h"


def main() -> None:
    fm = pd.read_csv(OUT / "finbert_match_predictions.csv")
    modern = pd.read_csv(OUT / "modern_predictions.csv")
    fm_eval = fm[fm["F2"].notna()]
    modern_eval = modern[modern["MODERN"].notna()]
    f2_error = float(np.max(np.abs(fm_eval["M_F2"] - fm_eval["F2"])))
    old = np.load(PRIVATE / "v1" / "modern_embeddings.npz")["embeddings"].astype(float)
    new = np.load(PRIVATE / "v2" / "modern_embeddings.npz")["embeddings"].astype(float)
    pooling_delta = np.linalg.norm(new - old, axis=1)
    inference = json.loads((OUT / "modern_inference.json").read_text())
    checks = {
        "canonical_finbert_reproduction_le_1e-10": f2_error <= 1e-10,
        "canonical_finbert_rows": len(fm_eval) == 1374,
        "modern_keys_and_rows_match": len(modern_eval) == len(fm_eval),
        "modern_pooling_excludes_special_tokens": inference["fingerprint"]["pooling"] == "special_token_excluded_mean",
        "modern_encoder_frozen": inference["trainable_parameters"] == 0 and not inference["gradients_present"],
        "pooling_changes_v1_vectors": bool(np.max(pooling_delta) > 1e-6),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "finbert_reproduction_max_abs_probability_error": f2_error,
        "modern_rows": len(modern_eval),
        "old_v1_vs_v2_pooling_mean_l2": float(pooling_delta.mean()),
        "old_v1_vs_v2_pooling_max_l2": float(pooling_delta.max()),
        "note": "Development and later are exposed exploratory historical backtests; v2 is a matched frozen-encoder probe, not a full Fin-ModernBERT reproduction.",
    }
    (OUT / "pooling_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
