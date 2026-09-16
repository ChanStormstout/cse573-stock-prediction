"""Fail-closed experiment provenance helpers. No ML imports or side effects."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def cache_readable(payload, metadata, expected):
    """Missing both means compute; partial, legacy or stale caches are errors."""
    payload, metadata = Path(payload), Path(metadata)
    if not payload.exists() and not metadata.exists():
        return False
    if not payload.exists() or not metadata.exists():
        raise RuntimeError("Partial cache: preserve it and use a new cache location.")
    meta = json.loads(metadata.read_text())
    if meta.get("fingerprint") != fingerprint(expected) or meta.get("payload_sha256") != sha(payload):
        raise RuntimeError("Legacy, changed or corrupted cache: preserve it; regenerate in a new location.")
    return True


def cache_metadata(payload, expected):
    return {"fingerprint": fingerprint(expected), "inputs_and_encoding": expected, "payload_sha256": sha(payload)}


def require_empty(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("Experiment directory already contains artifacts. Use a new version; no overwrite allowed.")


def verify_files(root, hashes):
    for name, digest in hashes.items():
        if sha(Path(root) / name) != digest:
            raise RuntimeError("Input/source fingerprint changed: " + str(name))


def pilot_gate(rows, n=20, threshold=.8):
    """Once first checkpoint fails, every resume still fails, even at n+1."""
    if len(rows) >= n:
        checkpoint = rows[:n]
        if any(sum(bool(r.get(metric, False)) for r in checkpoint) / n < threshold
               for metric in ("schema_valid", "verbatim_evidence")):
            raise RuntimeError("Predeclared quality stop persists: first 20 rows failed the 80% gate.")


def validate_resume(rows, expected_inputs, manifest, expected_fingerprint):
    pilot_gate(rows)
    if rows and (manifest is None or manifest.get("fingerprint") != expected_fingerprint):
        raise RuntimeError("Cannot resume legacy or changed pilot without matching input/prompt/model provenance.")
    if manifest is not None and manifest.get("fingerprint") != expected_fingerprint:
        raise RuntimeError("Pilot configuration changed; use a new experiment directory.")
    if len(rows) > len(expected_inputs):
        raise RuntimeError("More saved rows than expected inputs.")
    for i, row in enumerate(rows):
        expected = expected_inputs[i]
        if row.get("index") != i or any(row.get(k) != expected[k] for k in ("symbol", "record_key", "input_sha256")):
            raise RuntimeError("Pilot row does not match current ordered input: " + str(i))
