# Features utilisees par les modeles

Ce document decrit les features actuellement calculees dans le code et utilisees par la pipeline LSTM.

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

### Features triviales

1. `return_1d = log(C_t/C_{t-1})`  
Interpretation: capture le mouvement le plus recent du prix. Une valeur positive indique une dynamique haussiere immediate, souvent utilisee comme proxy de momentum tres court terme.

2. `return_2d = log(C_t/C_{t-2})`  
Interpretation: mesure l'evolution du prix sur deux jours consecutifs, permettant de lisser legerement le bruit journalier.

3. `return_5d = log(C_t/C_{t-5})`  
Interpretation: represente la dynamique hebdomadaire, utile pour detecter des tendances de court terme.

4. `return_10d = log(C_t/C_{t-10})`  
Interpretation: capture une tendance un peu plus installee sur environ deux semaines de trading.

5. `volatility_5d = std_5(return_1d)`  
Interpretation: mesure la dispersion recente des rendements. Une forte valeur indique un marche agite a court terme.

6. `volatility_10d = std_10(return_1d)`  
Interpretation: donne une vision intermediaire du risque, moins sensible aux fluctuations tres court terme.

7. `volatility_20d = std_20(return_1d)`  
Interpretation: proxy classique de volatilite mensuelle, souvent utilise pour caracteriser les regimes de marche.

8. `volume_change_1d = log(V_t/V_{t-1})`  
Interpretation: indique une acceleration ou un ralentissement du volume. Un pic de volume peut signaler un interet soudain du marche.

9. `volume_ratio_20d = V_t / mean_20(V)`  
Interpretation: compare le volume du jour a sa moyenne recente. Permet de detecter des jours atypiques.

10. `gap_1d = (O_t - C_{t-1}) / C_{t-1}`  
Interpretation: mesure le decalage entre la cloture precedente et l'ouverture du jour. Reflete l'information arrivee hors heures de marche.

11. `range_hl_1d = (H_t - L_t) / C_t`  
Interpretation: amplitude de la variation intraday. Une forte valeur traduit une grande incertitude ou volatilite intraday.

12. `open_to_close_1d = (C_t - O_t) / O_t`  
Interpretation: direction du mouvement pendant la session. Permet de distinguer les journees haussieres ou baissieres.

13. `clv = ((C_t-L_t) - (H_t-C_t)) / (H_t-L_t)`  
Interpretation: indique si le prix cloture proche du haut ou du bas de la journee. Reflète la pression acheteuse ou vendeuse.

14. `body_to_range = abs(C_t-O_t)/(H_t-L_t)`  
Interpretation: mesure la part du mouvement directionnel dans la bougie. Une valeur elevee indique un mouvement net sans forte indecision.

15. `upper_wick_ratio = (H_t-max(O_t,C_t))/(H_t-L_t)`  
Interpretation: indique un rejet des prix vers le bas apres avoir atteint un plus haut, souvent interprete comme un signal de pression vendeuse.

16. `lower_wick_ratio = (min(O_t,C_t)-L_t)/(H_t-L_t)`  
Interpretation: indique un rejet des prix vers le haut apres un plus bas, souvent interprete comme un signal de pression acheteuse.

17. `range_ratio_20 = (H_t-L_t)/mean_20(H-L)`  
Interpretation: compare la volatilite du jour a la volatilite moyenne recente. Permet d’identifier des phases d’expansion ou de contraction.

### Features exogenes (RF)

18. `vix_level = close_VIX_t`  
Interpretation: represente le niveau de volatilite implicite du marche. Un VIX eleve correspond a un contexte de stress ou d’incertitude.

---
## Pipeline LSTM (`src/lstm_pipeline/features.py`)

Le LSTM predit un log-return a horizon `h+1`:

- cible: `target = log(C_{t+h}/C_t)`

### Features de base (AAPL)

1. `log_return_1d = log(C_t/C_{t-1})`  
2. `log_return_5d = log(C_t/C_{t-5})`  
3. `log_return_20d = log(C_t/C_{t-20})`  
Interpretation: ces variables capturent le momentum a differents horizons, permettant au modele de detecter des tendances de court et moyen terme.

4. `ema_return_5 = EMA_5(log_return_1d)`  
5. `ema_return_20 = EMA_20(log_return_1d)`  
Interpretation: moyennes exponentielles des rendements, qui donnent plus de poids aux observations recentes et permettent de lisser le bruit.

6. `realized_vol_5d = sqrt(mean_5(r^2))`  
7. `realized_vol_20d = sqrt(mean_20(r^2))`  
Interpretation: mesures de volatilite realisee. Elles renseignent sur l’intensite des fluctuations recentes du marche.

8. `vol_ratio_5_20 = realized_vol_5d / realized_vol_20d`  
Interpretation: permet d’identifier les changements de regime de volatilite (par exemple passage d’un marche calme a un marche turbulent).

9. `gap_log = log(O_t/C_{t-1})`  
Interpretation: capture l’information overnight, souvent liee aux nouvelles ou evenements exterieurs.

10. `intraday_log = log(C_t/O_t)`  
Interpretation: mesure la performance intraday, utile pour distinguer les dynamiques internes a la session.

11. `volume_z_20 = (log(V_t) - mean_20(log(V))) / std_20(log(V))`  
Interpretation: detecte les anomalies de volume par rapport a la norme recente.

12. `day_of_week` (0=lundi ... 6=dimanche)  
Interpretation: capture d’eventuels effets calendaires (par exemple comportement different le lundi ou le vendredi).

13. `zscore_price_vs_ma20 = (C_t - MA20(C)) / (STD20(C)+eps)`  
Interpretation: mesure a quel point le prix est eloigne de sa moyenne mobile, normalise par sa volatilite.

14. `clv = ((C_t-L_t) - (H_t-C_t)) / (H_t-L_t)`  
15. `body_to_range = abs(C_t-O_t)/(H_t-L_t)`  
Interpretation: ces variables de micro-structure de bougie permettent de caracteriser le comportement du marche a l’echelle journaliere.

---

### Features regime optionnelles (`use_regime_features=true`)

16. `vol_ratio_5_60 = realized_vol_5d / realized_vol_60d`  
Interpretation: compare la volatilite recente a une volatilite de plus long terme, utile pour detecter les transitions de regime.

17. `rolling_skew_60 = skew_60(log_return_1d)`  
Interpretation: mesure l’asymetrie de la distribution des rendements. Une asymetrie peut indiquer des biais directionnels.

18. `rolling_kurtosis_60 = kurtosis_60(log_return_1d)`  
Interpretation: mesure l’epaisseur des queues de distribution, c’est-a-dire la frequence d’evenements extremes.

---

### Features exogenes optionnelles (`use_exog=true`)

19. `vix_level = close_VIX_t`  
Interpretation: niveau global de stress du marche.

20. `vix_change_1d = log(VIX_t/VIX_{t-1})`  
Interpretation: variation du stress de marche. Une hausse rapide du VIX correspond souvent a un choc de volatilite.

21. `vix_zscore_60 = (VIX_t - mean_60(VIX)) / (std_60(VIX)+eps)`  
Interpretation: indique si le VIX est anormalement eleve ou bas par rapport a son regime recent.

22. `spy_return_1d = log(SPY_t/SPY_{t-1})` (si SPY charge)  
Interpretation: capture la dynamique du marche global americain. Permet d’introduire une information de beta marche.

---

## Notes d'interpretation

- Features de returns:  
elles capturent la direction recente du marche mais restent tres bruitees et difficiles a exploiter seules.

- Features de volatilite:  
elles renseignent sur le niveau d’incertitude et sont essentielles pour detecter les changements de regime.

- Features intraday (CLV, body/range, wick):  
elles permettent de caracteriser la structure des bougies et de distinguer impulsion et rejet.

- Features exogenes (VIX/SPY):  
elles ajoutent un contexte macro au modele, indispensable pour comprendre les mouvements globaux du marche.
