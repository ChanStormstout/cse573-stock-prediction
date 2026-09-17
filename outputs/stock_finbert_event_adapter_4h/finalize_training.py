"""Add auditable timing and gradient summaries after the training driver exits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import PUBLIC, dump, sha


def main(run: Path) -> None:
    curves = json.loads((run / "training_curves.json").read_text())
    evidence = json.loads((run / "training_evidence.json").read_text())
    if len(curves) != 36 or evidence["runs"] != 36:
        raise ValueError({"expected_runs": 36, "curves": len(curves), "evidence": evidence["runs"]})
    summaries = {}
    for config in ("A0", "A1", "A2"):
        records = [row for row in curves if row["config"] == config]
        gradients = [epoch["gradient_norm_last"] for row in records for epoch in row["history"]]
        if not gradients or not all(math.isfinite(value) for value in gradients):
            raise ValueError({"nonfinite_or_missing_gradients": config})
        summaries[config] = {
            "runs": len(records),
            "epochs": sum(row["epochs_ran"] for row in records),
            "summed_run_seconds": sum(row["elapsed_seconds"] for row in records),
            "gradient_norm_preclip_last_batch_median": float(np.median(gradients)),
            "gradient_norm_preclip_last_batch_maximum": float(np.max(gradients)),
            "last_batches_above_clip_threshold": sum(value > 1.0 for value in gradients),
            "nonfinite_gradient_norms": 0,
        }
    evidence["per_config_training"] = summaries
    evidence["official_input"] = "protected candidate sentence plus all numbered target-company passage sentences; context-only truncation at 512 tokens"
    evidence["protocol_amendment_sha256"] = sha(PUBLIC / "PROTOCOL_AMENDMENT.md")
    evidence["cache_equivalence_sha256"] = sha(PUBLIC / "cache_equivalence.json")
    evidence["resume_provenance_sha256"] = sha(run / "resume_provenance.json")
    evidence["aborted_adjacent_context_run_used_for_selection"] = False
    source = Path(__file__).resolve().parent
    evidence["code_hashes"] = {name: sha(source / name) for name in ("adapter.py", "train_adapter.py", "finalize_training.py")}
    dump(run / "training_evidence.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.run)
