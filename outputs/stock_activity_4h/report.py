"""Reserved report entrypoint; it refuses unverified Activity v1 output."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main(root):
    verify=root/"verification.json"
    if not verify.exists() or json.loads(verify.read_text()).get("status")!="PASS": raise SystemExit("Activity report requires a PASS independent verifier.")
    raise SystemExit("Reserved for the later approved v1 result protocol.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);main(p.parse_args().root)
