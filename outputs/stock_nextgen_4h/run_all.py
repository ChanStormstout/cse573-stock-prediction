"""Configuration-driven prepare -> train/infer -> evaluate -> report entrypoint."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import B, ROOT


def call(*parts):
    command = [sys.executable, *map(str, parts)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(config: Path, start: str, stop: str, resume_llm: bool):
    values = json.loads(config.read_text())
    order = ["inventory", "paragraphs", "quality", "price", "llm", "calibration", "fusion", "cases", "publish", "verify"]
    if start not in order or stop not in order or order.index(start) > order.index(stop):
        raise ValueError("invalid stage range")
    selected = order[order.index(start) : order.index(stop) + 1]
    commands = {
        "inventory": (B / "inventory.py", "--out", values["inventory"]),
        "paragraphs": (B / "paragraph_inputs.py", "--out", values["paragraphs"]),
        "quality": (B / "input_quality.py", "--inputs", values["paragraphs"]),
        "price": (B / "recent_price.py", "--out", values["price"]),
        "llm": (B / "llm_choice.py", "--inputs", values["paragraphs"], "--out", values["llm"], "--batch-size", values["llm_batch_size"], *(["--resume"] if resume_llm else [])),
        "calibration": (B / "calibrate.py", "--paragraphs", values["paragraphs"], "--llm", values["llm"], "--price", values["price"], "--out", values["calibration"]),
        "fusion": (B / "fusion.py", "--price", values["price"], "--calibration", values["calibration"], "--out", values["fusion"]),
        "cases": (B / "cases.py", "--paragraphs", values["paragraphs"], "--calibration", values["calibration"], "--fusion", values["fusion"], "--out", values["cases"]),
        "publish": (B / "publish.py", "--inventory", values["inventory"], "--paragraphs", values["paragraphs"], "--llm", values["llm"], "--price", values["price"], "--calibration", values["calibration"], "--fusion", values["fusion"], "--cases", values["cases"], "--out", values["public"]),
        "verify": (B / "verify.py", "--work", str(Path(values["inventory"]).parent), "--public", values["public"]),
    }
    for stage in selected:
        call(*commands[stage])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=B / "config.json")
    parser.add_argument("--start", default="inventory")
    parser.add_argument("--stop", default="verify")
    parser.add_argument("--resume-llm", action="store_true")
    args = parser.parse_args()
    main(args.config, args.start, args.stop, args.resume_llm)
