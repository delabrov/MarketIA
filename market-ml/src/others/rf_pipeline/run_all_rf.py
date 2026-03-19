#!/usr/bin/env python3
# run_all_rf.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(title: str, cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 88)
    print(f"[run_all_rf] STEP: {title}")
    print("[run_all_rf] CMD :", " ".join(cmd))
    print("=" * 88 + "\n")
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_date", default=None, help="YYYY-MM-DD. If set, keep data >= start_date.")
    ap.add_argument("--leakage_audit", action="store_true", default=False)
    ap.add_argument("--leakage_smoke_model", action="store_true", default=False)
    args = ap.parse_args()

    py = sys.executable
    src_root = Path(__file__).resolve().parents[1]  # market-ml/src

    ticker = "AAPL"

    run_step("Download data", [py, "-m", "others.rf_pipeline.download_data", "--ticker", ticker], cwd=src_root)

    build_cmd = [py, "-m", "others.rf_pipeline.build_dataset", "--ticker", ticker]
    train_cmd = [py, "-m", "others.rf_pipeline.train_rf", "--ticker", ticker]
    eval_cmd = [py, "-m", "others.rf_pipeline.evaluate", "--ticker", ticker]
    if args.start_date:
        build_cmd += ["--start_date", args.start_date]
        train_cmd += ["--start_date", args.start_date]
        eval_cmd += ["--start_date", args.start_date]
    if args.leakage_audit:
        eval_cmd += ["--leakage_audit"]
    if args.leakage_smoke_model:
        eval_cmd += ["--leakage_smoke_model"]

    run_step("Build dataset", build_cmd, cwd=src_root)
    run_step("Train model", train_cmd, cwd=src_root)
    run_step("Evaluate + diagnostics + plots", eval_cmd, cwd=src_root)

    print("\n[run_all_rf] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
