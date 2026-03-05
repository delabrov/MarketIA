# LSTM Pipeline (AAPL Daily Log-Return)

## Run
From `market-ml/`:

```bash
python src/run_lstm.py --config configs/lstm.yaml
```

## Outputs
By default, outputs are written to:

```
results_lstm/<run_name>/
  models/
  plots/
  reports/
  preds/
  logs/
```

New diagnostics:
- `plots/<ticker>_feature_importance_permutation.png`
- `reports/<ticker>_feature_importance_permutation.csv`
- `plots/<ticker>_returns_true_vs_pred_test_h{h}.png`
- `plots/<ticker>_residuals_hist_test_h{h}.png`
- `plots/<ticker>_residuals_acf_test_h{h}.png`
- `plots/<ticker>_pred_vs_true_test_h{h}.png`
- `plots/<ticker>_decile_mean_true_h{h}.png`

## What it does
- Loads OHLCV from `data/raw/` (AAPL), and optionally `SPY` and `VIX` if present.
- Builds 15 base features (plus optional exogenous features).
- Builds sequences of length `sequence.length` and predicts next-day log-return.
- Trains LSTM with early stopping on validation loss.
- Evaluates on test, writes metrics, predictions, and plots.

## Config
See `configs/lstm.yaml` for:
- data selection
- features/exogenous toggles
- regime features toggle (`use_regime_features`)
- sequence length
- split ratios or dates
- model hyperparameters
- trading strategy settings (mode, threshold, costs)

Mono-horizon runs (multi-run supported):
- `prediction_horizon: 1` (single run)
- `run_horizons: [1, 3]` + `--multi-run` to run multiple horizons
- Outputs are isolated per run under `.../aapl_lstm_h{h}/`

Plots are generated per horizon:
- `plots/<ticker>_returns_true_vs_pred_test_h{h}.png`
- `plots/<ticker>_pred_vs_true_test_h{h}.png`
- `plots/<ticker>_residuals_hist_test_h{h}.png`
- `plots/<ticker>_residuals_acf_test_h{h}.png`
- `plots/<ticker>_decile_mean_true_h{h}.png`

Multi-horizon summary plots:
