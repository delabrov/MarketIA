# Features utilisees par les modeles

Cette section décrit les features calculees dans le code et utilisees par les pipelines LSTM.

## Conventions 

- `C_t`: close AAPL au jour `t`
- `O_t`: open AAPL au jour `t`
- `H_t`: high AAPL au jour `t`
- `L_t`: low AAPL au jour `t`
- `V_t`: volume AAPL au jour `t`
- `r_t = log(C_t / C_{t-1})`: log-return journalier
- `eps = 1e-12` quand un denominateur peut valoir zero

Hypotheses de construction:

- index temporel trie, en UTC
- fenetres rolling alignees sur `t` (pas de donnees futures)
- les NaN/inf dus aux fenetres (debut de serie) sont nettoyes via `dropna` final
- les series exogenes (VIX/SPY) sont alignees sur les dates du dataset (join sur l'index date)

## Pipeline LSTM (`src/lstm_pipeline/features.py`)

Le LSTM predit un log-return a horizon `h+1`:

- cible: `target = log(C_{t+h}/C_t)`

### Features de base (AAPL)

1. `log_return_1d = log(C_t/C_{t-1})`
2. `log_return_5d = log(C_t/C_{t-5})`
3. `log_return_20d = log(C_t/C_{t-20})`
Interpretation: momentum multi-horizons.

4. `ema_return_5 = EMA_5(log_return_1d)`
5. `ema_return_20 = EMA_20(log_return_1d)`
Interpretation: tendance lisse des retours.

6. `realized_vol_5d = sqrt(mean_5(r^2))`
7. `realized_vol_20d = sqrt(mean_20(r^2))`
Interpretation: volatilite realisee courte et mensuelle.

8. `vol_ratio_5_20 = realized_vol_5d / realized_vol_20d`
Interpretation: transition de regime de volatilite.

9. `gap_log = log(O_t/C_{t-1})`
Interpretation: information overnight.

10. `intraday_log = log(C_t/O_t)`
Interpretation: direction intraday.

11. `volume_z_20 = (log(V_t) - mean_20(log(V))) / std_20(log(V))`
Interpretation: surprise de volume.

12. `day_of_week` (0=lundi ... 6=dimanche)
Interpretation: effet calendaire.

13. `zscore_price_vs_ma20 = (C_t - MA20(C)) / (STD20(C)+eps)`
Interpretation: ecart normalise a la moyenne mobile.

14. `clv = ((C_t-L_t) - (H_t-C_t)) / (H_t-L_t)`
15. `body_to_range = abs(C_t-O_t)/(H_t-L_t)`
Interpretation: micro-structure de bougie.

### Features regime optionnelles (`use_regime_features=true`)

16. `vol_ratio_5_60 = realized_vol_5d / realized_vol_60d`
Interpretation: regime de risque court vs long.

17. `rolling_skew_60 = skew_60(log_return_1d)`
Interpretation: asymetrie de distribution recente.

18. `rolling_kurtosis_60 = kurtosis_60(log_return_1d)`
Interpretation: queues de distribution (fat tails).

### Features exogenes optionnelles (`use_exog=true`)

19. `vix_level = close_VIX_t`
Interpretation: niveau absolu de stress marche.

20. `vix_change_1d = log(VIX_t/VIX_{t-1})`
Interpretation: variation de stress (choc de volatilite implicite).

21. `vix_zscore_60 = (VIX_t - mean_60(VIX)) / (std_60(VIX)+eps)`
Interpretation: VIX anormalement haut/bas vs son regime recent.

22. `spy_return_1d = log(SPY_t/SPY_{t-1})` (si SPY charge)
Interpretation: beta marche broad US.

## Notes d'interpretation

- Features de returns:
elles capturent la direction/force recente mais peuvent etre tres bruites.

- Features de volatilite:
elles informent surtout sur l'incertitude et les changements de regime.

- Features intraday (CLV, body/range, wick):
elles mesurent la structure de la bougie, utile pour distinguer impulsion vs rejet.

- Features exogenes (VIX/SPY):
elles ajoutent le contexte macro-marche absent de la seule serie AAPL.

