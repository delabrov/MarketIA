#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from lstm_pipeline.run import main


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "configs" / "lstm.yaml"))
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--multi-run", action="store_true", default=False)
    args = ap.parse_args()
    raise SystemExit(main(Path(args.config), horizon=args.horizon, multi_run=args.multi_run))
