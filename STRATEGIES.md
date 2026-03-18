# 📈 Stratégies de Trading Basées sur un Modèle LSTM et un Overlay de Risque

## 🎯 Objectif

Ce projet vise à exploiter les prédictions d’un modèle LSTM appliqué à une série temporelle financière (ex : AAPL) afin de construire des stratégies d’investissement robustes.

L’approche repose sur deux piliers principaux :

1. **Un signal prédictif** issu du modèle (retour futur estimé)
2. **Un mécanisme de gestion du risque (overlay)** basé sur la volatilité réalisée

Je présente ici les stratégies finales retenues :

- `buyhold`
- `regime_overlay_longonly_base`
- `regime_overlay_condvol_highvol`
- `regime_overlay_condvol_p80`

---

# 🧠 1. Signal du modèle

Le modèle LSTM produit une prédiction de rendement futur :

\[
\hat{r}_{t+1}
\]

Ce signal est transformé en score standardisé (z-score) :

\[
z_t = \frac{\hat{r}_t - \mu}{\sigma}
\]

où :
- \( \mu \) = moyenne des prédictions (rolling ou globale)
- \( \sigma \) = écart-type des prédictions

Ce z-score permet de :

- détecter les signaux forts
- rendre le signal comparable dans le temps

---

# 🟦 2. Stratégie Buy & Hold

## 📌 Description

Stratégie de référence :

- exposition constante = 1
- aucun ajustement

\[
\text{exposure}_t = 1
\]

## 🎯 Rôle

- benchmark
- permet de mesurer la valeur ajoutée du modèle

---

# 🟧 3. regime_overlay_longonly_base

## 📌 Description

Stratégie directionnelle basée sur le signal LSTM.

### Règle :

\[
\text{exposure}_t =
\begin{cases}
1 & \text{si } z_t > 0 \\
0 & \text{sinon}
\end{cases}
\]

## 🧠 Interprétation

- on investit uniquement lorsque le modèle anticipe un rendement positif
- sinon, on reste en cash

## ⚠️ Limite

- aucune gestion du risque
- forte sensibilité aux faux signaux
- drawdowns encore élevés

---

# 🟢 4. regime_overlay_condvol_highvol

## 📌 Description

Ajout d’un **overlay de risque basé sur la volatilité**.

---

## 🔍 Calcul de la volatilité

Volatilité réalisée sur une fenêtre glissante :

\[
\sigma_t = \sqrt{252} \cdot \text{std}(r_{t-20:t})
\]

où :
- fenêtre de 20 jours
- annualisation

---

## ⚙️ Règle de décision

Définition d’un seuil fixe :

\[
\sigma_t > \theta \Rightarrow \text{régime de haute volatilité}
\]

---

## 🎯 Exposition

\[
\text{exposure}_t =
\begin{cases}
0.5 & \text{si volatilité élevée} \\
1.0 & \text{sinon}
\end{cases}
\]

---

## 🧠 Intuition

- en période de stress → réduction du risque
- en période calme → exposition normale

---

## ✅ Avantages

- réduction drastique des drawdowns
- volatilité plus faible
- meilleure stabilité

---

## ❌ Inconvénients

- coupe parfois trop tôt
- perte de performance en marché haussier

---

# 🔴 5. regime_overlay_condvol_p80

## 📌 Description

Version **adaptative** de la stratégie précédente.

Au lieu d’un seuil fixe, on utilise un **seuil dynamique basé sur un percentile**.

---

## 🔍 Calcul du seuil

On calcule le percentile sur une fenêtre longue (ex : 252 jours) :

\[
\theta_t = \text{percentile}_{80}(\sigma_{t-252:t})
\]

---

## ⚙️ Règle de décision

\[
\sigma_t > \theta_t \Rightarrow \text{volatilité élevée}
\]

---

## 🎯 Exposition

\[
\text{exposure}_t =
\begin{cases}
0.5 & \text{si } \sigma_t > \theta_t \\
1.0 & \text{sinon}
\end{cases}
\]

---

## 🧠 Intuition

- le seuil s’adapte au régime de marché
- robuste aux changements structurels
- meilleure calibration du risque

---

## ✅ Avantages

- meilleur compromis rendement / risque
- drawdown maîtrisé
- forte performance long terme

---

## ⚠️ Comparaison avec highvol

| Stratégie | Type de seuil | Réactivité |
|----------|-------------|------------|
| highvol | fixe | rigide |
| p80 | dynamique | adaptative |

---

# ⚖️ 6. Comparaison globale

## 📊 Synthèse qualitative

| Stratégie | Rendement | Risque | Robustesse |
|----------|----------|--------|-----------|
| buyhold | moyen | élevé | faible |
| longonly_base | élevé | élevé | moyen |
| condvol_highvol | moyen | faible | élevé |
| condvol_p80 | élevé | modéré | très élevé |

---

## 🏆 Conclusion

La stratégie :

\[
\boxed{\text{regime\_overlay\_condvol\_p80}}
\]

offre le meilleur compromis entre :

- performance
- contrôle du risque
- adaptabilité

---

# 🚀 7. Limites actuelles

Les stratégies actuelles restent :

- binaires (0.5 / 1.0)
- dépendantes de seuils discrets
- sensibles au timing

---

# 🔥 8. Pistes d’amélioration

## 1. Exposition continue

\[
\text{exposure}_t = f(\sigma_t)
\]

ex:
- faible vol → 1.2
- moyenne → 1.0
- élevée → 0.5

---

## 2. Intégration du signal modèle

pondérer par confiance du modèle :

\[
\text{exposure}_t = f(z_t, \sigma_t)
\]

---

## 3. Multi-actifs

- diversification
- réduction du risque global

---

## 4. Gestion des coûts

- intégrer les frais de transaction
- pénaliser le turnover

---

# 🧠 9. Conclusion finale

Ce framework montre qu’un simple modèle prédictif combiné à :

- un filtrage directionnel
- un overlay de volatilité adaptatif

permet de :

- battre le buy & hold
- réduire significativement les drawdowns
- améliorer le Sharpe ratio

👉 La clé n’est pas uniquement le modèle, mais **la gestion du risque**.

---
