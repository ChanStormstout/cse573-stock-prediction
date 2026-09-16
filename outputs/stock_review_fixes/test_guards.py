import json
from pathlib import Path
import tempfile
import unittest
from guards import cache_readable, cache_metadata, fingerprint, pilot_gate, require_empty, validate_resume, verify_files, sha


class GuardTests(unittest.TestCase):
    def test_restart_after_failed_checkpoint_at_20_or_21(self):
        rows = [dict(schema_valid=i < 11, verbatim_evidence=i < 10) for i in range(20)]
        for r in (rows, rows + [dict(schema_valid=True, verbatim_evidence=True)]):
            with self.assertRaisesRegex(RuntimeError, "quality stop"):
                pilot_gate(r)

    def test_resume_checks_order_input_and_config(self):
        inp = [dict(symbol="AAPL", record_key="a::b", input_sha256="abc")]
        rows = [dict(index=0, **inp[0])]
        validate_resume(rows, inp, {"fingerprint": "f"}, "f")
        for manifest in (None, {"fingerprint": "old"}):
            with self.assertRaises(RuntimeError): validate_resume(rows, inp, manifest, "f")
        with self.assertRaises(RuntimeError): validate_resume(rows, [dict(inp[0], input_sha256="changed")], {"fingerprint": "f"}, "f")
        with self.assertRaises(RuntimeError): validate_resume([dict(rows[0], index=1)], inp, {"fingerprint": "f"}, "f")

    def test_cache_rejects_changed_inputs_config_or_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, m = Path(tmp)/"cache", Path(tmp)/"meta.json"
            expected = dict(title="old", upstream="u", revision="r", max_length=256, pooling="mean")
            self.assertFalse(cache_readable(p,m,expected))
            p.write_text("features")
            with self.assertRaises(RuntimeError): cache_readable(p,m,expected)
            m.write_text(json.dumps(cache_metadata(p,expected)))
            self.assertTrue(cache_readable(p,m,expected))
            for k in expected:
                with self.assertRaises(RuntimeError): cache_readable(p,m,dict(expected, **{k:"changed"}))
            p.write_text("corrupted")
            with self.assertRaises(RuntimeError): cache_readable(p,m,expected)

    def test_directory_and_training_source_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);require_empty(root);p=root/"input";p.write_text("original")
            hashes={"input":sha(p)};verify_files(root,hashes)
            with self.assertRaises(RuntimeError): require_empty(root)
            p.write_text("changed")
            with self.assertRaises(RuntimeError): verify_files(root,hashes)

    def test_fingerprint_order_and_values(self):
        self.assertEqual(fingerprint({"a":1,"b":2}),fingerprint({"b":2,"a":1}))
        self.assertNotEqual(fingerprint({"a":1}),fingerprint({"a":2}))


if __name__ == "__main__": unittest.main()
