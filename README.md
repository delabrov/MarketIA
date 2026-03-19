# Prédiction de séries temporelles financières avec LSTM

## Objectif du projet

Ce projet a pour objectif de prédire l’évolution à court terme du prix d’un actif financier (ici Apple, AAPL) à l’aide d’un modèle de type LSTM (Long Short-Term Memory). Plus précisément, on cherche à estimer le **rendement du lendemain (retour à J+1)**, c’est-à-dire la variation du prix entre aujourd’hui et le jour suivant.

À partir de cette prédiction, l’idée est ensuite de construire une **stratégie d’investissement** : autrement dit, décider quand investir, quand réduire son exposition, et comment gérer le risque.

Ce projet s’inscrit alors à l’intersection de deux domaines :
- le **machine learning**, pour faire des prédictions
- la **finance quantitative**, pour transformer ces prédictions en décisions d’investissement

---

## Comprendre le problème

Les marchés financiers évoluent dans le temps, et leurs dynamiques sont souvent complexes, bruitées et difficiles à modéliser. On parle de **série temporelle (time series)**, c’est-à-dire une suite de données ordonnées dans le temps.

Plutôt que d’essayer de prédire directement le prix, ce projet se concentre sur le **rendement (return)**, qui correspond au pourcentage de variation du prix entre deux instants. Ce choix permet de travailler avec des données plus stables et comparables.

Le défi est donc le suivant :

> À partir des informations passées, peut-on prédire le rendement du lendemain ?

---

## Pourquoi un LSTM ?

Le modèle utilisé ici est un **LSTM (Long Short-Term Memory)**, un type particulier de réseau de neurones conçu pour traiter des séquences.

Contrairement à des modèles classiques, un LSTM est capable de :
- prendre en compte le **contexte temporel**,
- mémoriser des informations importantes dans le passé,
- oublier les informations non pertinentes.

En d’autres termes, il est particulièrement adapté aux données où **l’ordre dans le temps est essentiel**, comme les séries financières.

---

## Construction des données

Avant d’entraîner le modèle, il est nécessaire de transformer les données brutes en variables exploitables : le **feature engineering**.

À partir des prix historiques, plusieurs types d’indicateurs sont construits :

- des indicateurs de tendance (ex : moyennes mobiles),
- des indicateurs de volatilité (mesure des variations du marché),
- des indicateurs de momentum (vitesse des mouvements de prix).

Ces variables, appelées **features** par la suite ([FEATURES.md](FEATURES.md)), permettent de donner au modèle une vision plus riche du marché que le simple prix.

Les données sont ensuite normalisées (mise à l’échelle) afin de faciliter l’apprentissage du modèle.

---
## Entraînement du modèle

Le modèle utilisé dans ce projet est un **LSTM (Long Short-Term Memory)**, conçu pour traiter des données séquentielles. Contrairement à un modèle classique qui prend une observation indépendante, le LSTM reçoit ici une **séquence temporelle** de plusieurs jours consécutifs.

Dans notre cas, chaque jour t est représenté par un vecteur de variables (features) construites à partir des données de marché (prix, volume, volatilité, etc.). Plutôt que d’utiliser uniquement les données du jour t, le modèle reçoit une fenêtre glissante de longueur L (par exemple 60 jours) :

X_t = (x_{t-L+1}, ..., x_t)

Cette séquence permet au modèle de capturer des dynamiques temporelles comme :
- des tendances (momentum),
- des changements de volatilité,
- des phases de régime (marché calme vs stress).

La cible du modèle est le **log-return à horizon 1 jour** :

y_t = log(C_{t+1} / C_t)

Le modèle apprend donc à approximer une fonction :

y_hat_t = f(X_t)

où X_t contient l’historique des features sur les derniers jours.

---

### Fonctionnement du LSTM dans ce projet

Le LSTM traite la séquence jour par jour et maintient un **état interne (memory)** qui évolue dans le temps. Cet état lui permet de retenir certaines informations importantes (par exemple un changement de volatilité) et d’en oublier d’autres.

Concrètement, le modèle apprend à détecter des patterns temporels dans les features, comme :
- une augmentation progressive de la volatilité,
- une phase de surachat / survente,
- ou une dynamique de momentum.

L’information n’est donc pas seulement contenue dans les features à un instant donné, mais dans leur **évolution dans le temps**.

---

### Construction des données

La construction du dataset suit des contraintes strictes pour éviter toute fuite d’information (data leakage) :

- toutes les features sont calculées uniquement avec des données passées (rolling windows alignées sur t),
- la cible utilise uniquement le futur immédiat (t+1),
- les données sont séparées chronologiquement en train / validation / test.

Aucun mélange aléatoire (shuffle) n’est effectué.

---

### Sélection des features (point clé du projet)

Un point important du projet est que les features n’ont pas été choisies arbitrairement.

Une procédure de **feature ablation** ([ABLATION.md](ABLATION.md)) a été utilisée :

- le modèle est entraîné avec toutes les features,
- puis on retire une feature à la fois,
- on mesure l’impact sur les performances (IC, RMSE),
- on conserve uniquement les features qui apportent réellement de l’information.

Cela permet d’éviter :
- les variables inutiles,
- la redondance,
- et le bruit dans le modèle.

Le modèle final utilise donc un ensemble de features **empiriquement validé**, et non simplement “intuitif”.

---

### Entraînement

Le modèle est entraîné en minimisant une erreur de type MSE entre la prédiction et le retour réel.

Un mécanisme d’**early stopping** est utilisé pour éviter l’overfitting :
l’entraînement s’arrête lorsque la performance sur l’ensemble de validation ne s’améliore plus.

Les données sont normalisées à partir du train uniquement, puis appliquées à la validation et au test.

---

## Évaluation des performances

L’évaluation du modèle ne repose pas uniquement sur l’erreur de prédiction. En finance, ce qui compte est la **qualité du signal**, pas seulement sa précision brute.

---

### Métriques utilisées

Plusieurs métriques sont utilisées dans le projet :

#### RMSE / MAE

Mesurent l’erreur moyenne entre les prédictions et les valeurs réelles.

Elles permettent de vérifier que le modèle ne diverge pas, mais restent secondaires.

---

#### Information Coefficient (IC)

IC = corr(y_hat, y)

C’est la corrélation entre les prédictions et les retours réels.

C’est la métrique la plus importante du projet :
elle mesure directement si le modèle capte un signal exploitable.

---

#### Rank IC

Corrélation de rang entre les prédictions et les retours.

Permet de vérifier si le modèle classe correctement les situations, même si les amplitudes sont imparfaites.

---

#### Hit Ratio

Proportion de fois où le modèle prédit correctement le signe du retour.

Même un hit ratio proche de 52–53% peut être exploitable.

---

### Analyse par déciles

Les prédictions sont triées et découpées en groupes (déciles).

On observe ensuite le retour moyen réel dans chaque groupe.

Un bon modèle doit produire une relation monotone :
les meilleures prédictions doivent correspondre aux meilleurs retours.

Cette analyse est centrale pour vérifier que le modèle produit un signal exploitable.

---

### Analyse des résidus

Les résidus (erreur entre réel et prédit) sont analysés pour détecter :

- un biais systématique,
- une structure non capturée,
- une autocorrélation restante.

Cela permet d’identifier les limites du modèle.

---

### Importance des features

La contribution de chaque feature est évaluée via :

- permutation importance,
- feature ablation (ΔIC, ΔRMSE).

Cela permet de comprendre quelles variables portent réellement le signal.

---

## Objectif réel

Le but du modèle n’est pas de prédire parfaitement les rendements.

Le but est de produire un signal :

- légèrement corrélé au futur,
- stable dans le temps,
- exploitable dans une stratégie.

C’est cette combinaison entre **modèle + sélection de variables + évaluation rigoureuse** qui permet d’obtenir un système cohérent.

## De la prédiction à la stratégie

Le modèle produit un signal : une estimation du rendement futur.

Ce signal est ensuite utilisé pour construire une stratégie simple :
- si le rendement prédit est positif → on investit
- s’il est négatif → on réduit ou coupe l’exposition

Ce mécanisme permet de transformer une prédiction en **règle de décision**.

Ensuite, des mécanismes de gestion du risque peuvent être ajoutés pour améliorer la stabilité de la stratégie, notamment en tenant compte de la volatilité du marché.

---

## Organisation du projet

Le projet est structuré en plusieurs étapes :

1. **Collecte et préparation des données**
2. **Construction des features**
3. **Entraînement du modèle LSTM**
4. **Génération des prédictions**
5. **Backtest de stratégies d’investissement**

Chaque étape est pensée pour être reproductible et modulaire, afin de pouvoir tester facilement de nouvelles idées.

---

## Conclusion

Ce projet montre comment un modèle de machine learning, en particulier un LSTM, peut être utilisé pour analyser des séries temporelles financières et générer des signaux exploitables.

Il met en évidence un point essentiel :

> La valeur ne vient pas uniquement de la prédiction, mais de la manière dont elle est utilisée pour prendre des décisions.

C’est cette combinaison entre **modélisation et stratégie** qui permet de construire des approches robustes et performantes.

---
