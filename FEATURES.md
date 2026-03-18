# Features utilisees par les modeles

Ce document decrit les features actuellement calculees dans le code et utilisees par les pipelines RF et LSTM.

## Conventions communes

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

## Pipeline RF (`src/rf_pipeline/build_dataset.py`)

Le modele RF predit `target_up`, derive de `target_next_log_return = log(C_{t+1}/C_t)`.

### Features triviales

1. `return_1d = log(C_t/C_{t-1})`
Interpretation: momentum tres court terme.

2. `return_2d = log(C_t/C_{t-2})`
Interpretation: mouvement sur 2 jours.

3. `return_5d = log(C_t/C_{t-5})`
Interpretation: dynamique hebdo.

4. `return_10d = log(C_t/C_{t-10})`
Interpretation: dynamique 2 semaines.

5. `volatility_5d = std_5(return_1d)`
Interpretation: risque recent (court).

6. `volatility_10d = std_10(return_1d)`
Interpretation: risque court/moyen terme.

7. `volatility_20d = std_20(return_1d)`
Interpretation: regime de volatilite mensuel.

8. `volume_change_1d = log(V_t/V_{t-1})`
Interpretation: acceleration/frein du flux de volume.

9. `volume_ratio_20d = V_t / mean_20(V)`
Interpretation: volume du jour versus normalite recente.

10. `gap_1d = (O_t - C_{t-1}) / C_{t-1}`
Interpretation: choc overnight.

11. `range_hl_1d = (H_t - L_t) / C_t`
Interpretation: amplitude intraday.

12. `open_to_close_1d = (C_t - O_t) / O_t`
Interpretation: direction de la session cash.

13. `clv = ((C_t-L_t) - (H_t-C_t)) / (H_t-L_t)`
Interpretation: position de cloture dans la bougie (pression acheteuse/vendeuse).

14. `body_to_range = abs(C_t-O_t)/(H_t-L_t)`
Interpretation: part de mouvement directionnel dans la range.

15. `upper_wick_ratio = (H_t-max(O_t,C_t))/(H_t-L_t)`
Interpretation: rejet haussier intraday.

16. `lower_wick_ratio = (min(O_t,C_t)-L_t)/(H_t-L_t)`
Interpretation: rejet baissier intraday.

17. `range_ratio_20 = (H_t-L_t)/mean_20(H-L)`
Interpretation: compression/expansion de range.

### Features exogenes (RF)

18. `vix_level = close_VIX_t`
Interpretation: niveau de stress/aversion au risque du marche.

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

## Notes d interpretation

- Features de returns:
elles capturent la direction/force recente mais peuvent etre tres bruites.

- Features de volatilite:
elles informent surtout sur l'incertitude et les changements de regime.

- Features intraday (CLV, body/range, wick):
elles mesurent la structure de la bougie, utile pour distinguer impulsion vs rejet.

- Features exogenes (VIX/SPY):
elles ajoutent le contexte macro-marche absent de la seule serie AAPL.

