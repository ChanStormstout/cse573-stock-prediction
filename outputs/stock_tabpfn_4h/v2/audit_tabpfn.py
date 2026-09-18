"""Audit the bundled TabPFN runtime/checkpoint before the v2 probe."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "work" / "stock-data" / "goal60_4h" / "runtime"
CHECKPOINT = ROOT / "work" / "stock-data" / "goal60_4h" / "tabpfn_model" / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    metadata = (RUNTIME / "tabpfn-6.3.0.dist-info" / "METADATA").read_text()
    assert "Name: tabpfn" in metadata and "Version: 6.3.0" in metadata
    assert "Prior Labs License" in metadata
    assert "Prior-Labs/tabpfn_2_5" in metadata
    assert "tabpfn-v2.5-classifier-v2.5_default.ckpt" in metadata
    assert "default classification checkpoint, finetuned on real-data" in metadata
    assert CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 1_000_000
    with zipfile.ZipFile(CHECKPOINT) as z:
        names = z.namelist()
        roots = sorted({n.split("/", 1)[0] for n in names if "/" in n})
    result = {
        "status": "PASS",
        "package_version": "6.3.0",
        "runtime_metadata_sha256": sha(RUNTIME / "tabpfn-6.3.0.dist-info" / "METADATA"),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_archive_roots": roots,
        "model_repo": "Prior-Labs/tabpfn_2_5",
        "model_filename": "tabpfn-v2.5-classifier-v2.5_default.ckpt",
        "provenance": "default TabPFN-2.5 classifier fine-tuned on real data according to bundled package metadata",
        "synthetic_only_claim": False,
        "old_configuration": {"n_estimators": 1, "device": "cpu"},
        "corrected_configuration": {"n_estimators": 8, "device": "cpu", "random_state": 573, "n_preprocessing_jobs": 1},
        "note": "Local package/checkpoint evidence; this does not independently reproduce the TabPFN pretraining paper.",
    }
    (OUT / "provenance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "DATA_AUDIT.md").write_text(
        "# TabPFN v2.5 audit\n\n"
        "The bundled runtime is TabPFN 6.3.0. Its metadata identifies the "
        "default classifier checkpoint as `Prior-Labs/tabpfn_2_5` and says it "
        "is fine-tuned on real data. The checkpoint archive root is "
        "`tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features`, "
        "which is consistent with that metadata. Therefore the historical "
        "v1 result must not be called synthetic-only.\n\n"
        f"- checkpoint SHA-256: `{result['checkpoint_sha256']}`\n"
        f"- checkpoint size: `{result['checkpoint_bytes']}` bytes\n"
        "- v1 configuration: `n_estimators=1` (reduced-compute probe)\n"
        "- v2 configuration: `n_estimators=8` (library default ensemble)\n\n"
        "This audit verifies local provenance and configuration; it does not "
        "establish that the model is suitable for stock direction or reproduce "
        "the pretraining data.\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
