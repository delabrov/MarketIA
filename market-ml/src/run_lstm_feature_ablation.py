#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None
import json

from lstm_pipeline.feature_ablation import (
    run_ablation,
    save_summary,
    plot_delta_barh,
    plot_delta_scatter,
)


def load_config(path: Path) -> dict:
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Please add pyyaml to requirements.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "lstm.yaml"),
    )
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--mode", type=str, default="drop", choices=["drop", "add"])
    ap.add_argument("--features", type=str, default="")
    ap.add_argument("--core_features", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_repeats", type=int, default=1)
    ap.add_argument("--max_features_tested", type=int, default=None)
    ap.add_argument("--save_models", action="store_true", default=False)
    args = ap.parse_args()

    cfg = load_config(Path(args.base_config))

    features_subset = [f.strip() for f in args.features.split(",") if f.strip()] if args.features else None
    core_features = [f.strip() for f in args.core_features.split(",") if f.strip()] if args.core_features else None

    ticker = cfg.get("data", {}).get("ticker", "AAPL").lower()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg.get("outputs", {}).get("root", "results_lstm"))
    run_dir = out_root / "feature_ablation" / f"{ticker}_h{args.horizon}" / timestamp
    reports_dir = run_dir / "reports"
    plots_dir = run_dir / "plots"

    print(f"[ablation] Starting feature ablation: mode={args.mode}, horizon={args.horizon}")
    print(f"[ablation] Output dir: {run_dir}")

    df_out, summary = run_ablation(
        cfg,
        horizon=args.horizon,
        mode=args.mode,
        features_subset=features_subset,
        seed=args.seed,
        n_repeats=args.n_repeats,
        max_features_tested=args.max_features_tested,
        save_models=args.save_models,
        output_root=out_root,
        core_features=core_features,
    )

    summary["timestamp_utc"] = timestamp
    if df_out is not None and len(df_out) > 0:
        top_pos = df_out.sort_values("delta_ic", ascending=False).head(10)
        top_neg = df_out.sort_values("delta_ic", ascending=True).head(10)
        summary["top_10_useful_delta_ic"] = top_pos[["feature", "delta_ic", "delta_rmse"]].to_dict(orient="records")
        summary["top_10_nuisible_delta_ic"] = top_neg[["feature", "delta_ic", "delta_rmse"]].to_dict(orient="records")

    base = summary.get("baseline", {})
    if base:
        print(
            "[ablation] Baseline: "
            f"IC={base.get('ic', float('nan')):.4f}, "
            f"RMSE={base.get('rmse', float('nan')):.6f}, "
            f"MAE={base.get('mae', float('nan')):.6f}"
        )

    save_summary(df_out, summary, reports_dir)

    plot_delta_barh(
        df_out,
        value_col="delta_ic",
        save_path=plots_dir / "delta_ic_sorted.png",
        title="Feature Ablation: ΔIC (base - ablated)",
    )
    plot_delta_barh(
        df_out,
        value_col="delta_rmse",
        save_path=plots_dir / "delta_rmse_sorted.png",
        title="Feature Ablation: ΔRMSE (ablated - base)",
    )
    plot_delta_scatter(
        df_out,
        save_path=plots_dir / "delta_ic_vs_delta_rmse.png",
        title="Feature Ablation: ΔIC vs ΔRMSE",
    )

    print(f"[ablation] Saved results: {reports_dir / 'feature_ablation_results.csv'}")
    print(f"[ablation] Saved summary: {reports_dir / 'feature_ablation_summary.json'}")
    print(f"[ablation] Saved plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
