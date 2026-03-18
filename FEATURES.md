# Features utilisees par les modeles

Ce document décrit les features actuellement calculées dans le code et utilisées par la pipeline LSTM.

## Conventions

- `C_t`: close AAPL au jour `t`
- `O_t`: open AAPL au jour `t`
- `H_t`: high AAPL au jour `t`
- `L_t`: low AAPL au jour `t`
- `V_t`: volume AAPL au jour `t`
- `r_t = log(C_t / C_{t-1})`: log-return journalier
- `eps = 1e-12` quand un denominateur peut valoir zéro

Hypotheses de construction:

- index temporel trié, en UTC
- fenêtres rolling alignées sur `t` (pas de données futures)
- les NaN/inf dus aux fenêtres (debut de serie) sont nettoyés via `dropna` final
- les series exogènes (VIX/SPY) sont alignées sur les dates du dataset (join sur l'index date)

### Features triviales

1. `return_1d = log(C_t/C_{t-1})`  
Interprétation: capture le mouvement le plus recent du prix. Une valeur positive indique une dynamique haussiere immediate, souvent utilisée comme proxy de momentum tres court terme.

2. `return_2d = log(C_t/C_{t-2})`  
Interprétation: mesure l'évolution du prix sur deux jours consécutifs, permettant de lisser légèrement le bruit journalier.

3. `return_5d = log(C_t/C_{t-5})`  
Interprétation: représente la dynamique hebdomadaire, utile pour détecter des tendances de court terme.

4. `return_10d = log(C_t/C_{t-10})`  
Interpéetation: capture une tendance un peu plus installée sur environ deux semaines de trading.

5. `volatility_5d = std_5(return_1d)`  
Interprétation: mesure la dispersion récente des rendements. Une forte valeur indique un marche agité à court terme.

6. `volatility_10d = std_10(return_1d)`  
Interprétation: donne une vision intermédiaire du risque, moins sensible aux fluctuations très court terme.

7. `volatility_20d = std_20(return_1d)`  
Interprétation: proxy classique de volatilite mensuelle, souvent utilisé pour caracteriser les régimes de marché.

8. `volume_change_1d = log(V_t/V_{t-1})`  
Interprétation: indique une acceleration ou un ralentissement du volume. Un pic de volume peut signaler un intérêt soudain du marché.

9. `volume_ratio_20d = V_t / mean_20(V)`  
Interprétation: compare le volume du jour à sa moyenne récente. Permet de détecter des jours atypiques.

10. `gap_1d = (O_t - C_{t-1}) / C_{t-1}`  
Interprétation: mesure le décalage entre la cloture précedente et l'ouverture du jour. Reflète l'information arrivée hors heures de marché.

11. `range_hl_1d = (H_t - L_t) / C_t`  
Interprétation: amplitude de la variation intraday. Une forte valeur traduit une grande incertitude ou volatilité intraday.

12. `open_to_close_1d = (C_t - O_t) / O_t`  
Interprétation: direction du mouvement pendant la session. Permet de distinguer les journées haussieres ou baissieres.

13. `clv = ((C_t-L_t) - (H_t-C_t)) / (H_t-L_t)`  
Interprétation: indique si le prix cloture proche du haut ou du bas de la journee. Reflète la pression acheteuse ou vendeuse.

14. `body_to_range = abs(C_t-O_t)/(H_t-L_t)`  
Interprétation: mesure la part du mouvement directionnel dans la bougie. Une valeur élevée indique un mouvement net sans forte indecision.

15. `upper_wick_ratio = (H_t-max(O_t,C_t))/(H_t-L_t)`  
Interpretation: indique un rejet des prix vers le bas apres avoir atteint un plus haut, souvent interpreté comme un signal de pression vendeuse.

16. `lower_wick_ratio = (min(O_t,C_t)-L_t)/(H_t-L_t)`  
Interprétation: indique un rejet des prix vers le haut apres un plus bas, souvent interpreté comme un signal de pression acheteuse.

17. `range_ratio_20 = (H_t-L_t)/mean_20(H-L)`  
Interprétation: compare la volatilite du jour a la volatilité moyenne recente. Permet d’identifier des phases d’expansion ou de contraction.

### Features exogenes (RF)

18. `vix_level = close_VIX_t`  
Interpretation: represente le niveau de volatilite implicite du marche. Un VIX élevé correspond à un contexte de stress ou d’incertitude.

---
## Pipeline LSTM (`src/lstm_pipeline/features.py`)

Le LSTM predit un log-return a horizon `h+1`:

- cible: `target = log(C_{t+h}/C_t)`

### Features de base (AAPL)

1. `log_return_1d = log(C_t/C_{t-1})`  
2. `log_return_5d = log(C_t/C_{t-5})`  
3. `log_return_20d = log(C_t/C_{t-20})`  
Interprétation: ces variables capturent le momentum a différents horizons, permettant au modèle de détecter des tendances de court et moyen terme.

4. `ema_return_5 = EMA_5(log_return_1d)`  
5. `ema_return_20 = EMA_20(log_return_1d)`  
Interprétation: moyennes exponentielles des rendements, qui permettent de lisser le bruit.

6. `realized_vol_5d = sqrt(mean_5(r^2))`  
7. `realized_vol_20d = sqrt(mean_20(r^2))`  
Interprétation: mesures de volatilité réalisée. Elles renseignent sur l’intensité des fluctuations récentes du marché.

8. `vol_ratio_5_20 = realized_vol_5d / realized_vol_20d`  
Interprétation: permet d’identifier les changements de régime de volatilité (par exemple passage d’un marché calme a un marché turbulent).

9. `gap_log = log(O_t/C_{t-1})`  
Interpretation: capture l’information overnight, souvent liée aux nouvelles ou évenements exterieurs.

10. `intraday_log = log(C_t/O_t)`  
Interpretation: mesure la performance intraday, utile pour distinguer les dynamiques internes à la session.

11. `volume_z_20 = (log(V_t) - mean_20(log(V))) / std_20(log(V))`  
Interprétation: détecte les anomalies de volume par rapport à la norme récente.

12. `day_of_week` (0=lundi ... 6=dimanche)  
Interprétation: capture d’éventuels effets calendaires (par exemple comportement different le lundi ou le vendredi).

13. `zscore_price_vs_ma20 = (C_t - MA20(C)) / (STD20(C)+eps)`  
Interprétation: mesure à quel point le prix est éloigné de sa moyenne mobile, normalise par sa volatilite.

14. `clv = ((C_t-L_t) - (H_t-C_t)) / (H_t-L_t)`  
15. `body_to_range = abs(C_t-O_t)/(H_t-L_t)`  
Interprétation: ces variables de micro-structure de bougie permettent de caracteriser le comportement du marche a l’echelle journaliere.

---

### Features regime optionnelles (`use_regime_features=true`)

16. `vol_ratio_5_60 = realized_vol_5d / realized_vol_60d`  
Interprétation: compare la volatilité récente à une volatilité de plus long terme, utile pour détecter les transitions de régime.

17. `rolling_skew_60 = skew_60(log_return_1d)`  
Interprétation: mesure l’asymétrie de la distribution des rendements. Une asymetrie peut indiquer des biais directionnels.

18. `rolling_kurtosis_60 = kurtosis_60(log_return_1d)`  
Interprétation: mesure l’épaisseur des queues de distribution, c’est-à-dire la fréquence d’évenements extrêmes.

---

### Features exogenes optionnelles (`use_exog=true`)

19. `vix_level = close_VIX_t`  
Interpretation: niveau global de stress du marche.

20. `vix_change_1d = log(VIX_t/VIX_{t-1})`  
Interpretation: variation du stress de marche. Une hausse rapide du VIX correspond souvent à un choc de volatilité.

21. `vix_zscore_60 = (VIX_t - mean_60(VIX)) / (std_60(VIX)+eps)`  
Interprétation: indique si le VIX est anormalement élevé ou bas par rapport à son regime récent.

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
