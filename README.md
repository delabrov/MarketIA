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

Le cœur du projet repose sur un modèle **LSTM** (*Long Short-Term Memory*), c’est-à-dire un réseau de neurones récurrent (*Recurrent Neural Network*, ou **RNN**) conçu pour traiter des données ordonnées dans le temps. Contrairement à un modèle tabulaire classique, qui considère chaque observation comme indépendante, un LSTM exploite explicitement le fait que les données financières sont des **séries temporelles (time series)**. L’objectif est donc de fournir au modèle non pas une seule ligne de données, mais une **séquence** de plusieurs jours consécutifs, afin qu’il apprenne à détecter des dynamiques, des motifs récurrents et des changements de régime.

Dans ce projet, chaque date \(t\) est décrite par un vecteur de variables \(x_t \in \mathbb{R}^d\), où \(d\) est le nombre de **features** utilisées. Ces features sont construites à partir de l’historique de prix, de volume, de volatilité et éventuellement de variables exogènes comme le VIX. Le modèle ne regarde pas uniquement la date \(t\), mais une **fenêtre temporelle** de longueur \(L\), par exemple \(L = 60\) jours. Pour chaque date \(t\), l’entrée du modèle est donc :

\[
X_t = (x_{t-L+1}, x_{t-L+2}, \dots, x_t)
\]

Cette matrice d’entrée a une dimension \((L, d)\). Elle représente l’état du marché sur les \(L\) derniers jours.

La cible, elle, est un **rendement futur**. Dans le cas du projet, on prédit le plus souvent le rendement logarithmique à horizon \(h=1\), c’est-à-dire :

\[
y_t = \log\left(\frac{C_{t+1}}{C_t}\right)
\]

où \(C_t\) désigne le prix de clôture (*close price*) au jour \(t\). Ce choix est important : on ne prédit pas directement le prix futur, mais le **retour** (*return*), car les prix sont non stationnaires alors que les retours sont en général plus adaptés à l’apprentissage statistique.

Le modèle apprend donc une fonction de la forme :

\[
\hat{y}_t = f_\theta(X_t)
\]

où \(\theta\) représente l’ensemble des paramètres du réseau. L’idée est que la prédiction du retour futur dépend non seulement de l’état actuel du marché, mais aussi de l’évolution récente de plusieurs indicateurs.

### Pourquoi utiliser un LSTM ?

Un RNN classique met à jour un état caché \(h_t\) à partir de l’entrée \(x_t\) et de l’état précédent \(h_{t-1}\). En simplifiant, cela s’écrit :

\[
h_t = \phi(W_x x_t + W_h h_{t-1} + b)
\]

Mais ce type de modèle souffre rapidement du problème de **vanishing gradient**, c’est-à-dire que l’information lointaine devient difficile à conserver pendant l’apprentissage. Le LSTM a été conçu précisément pour résoudre ce problème en introduisant une **mémoire interne** \(c_t\) et plusieurs **portes (gates)** qui contrôlent ce qui doit être retenu ou oublié.

À chaque pas de temps \(t\), le LSTM calcule :

- une **forget gate** \(f_t\), qui décide quelle part de l’ancienne mémoire doit être conservée ;
- une **input gate** \(i_t\), qui décide quelle nouvelle information doit entrer dans la mémoire ;
- une **candidate memory** \(\tilde{c}_t\), c’est-à-dire la nouvelle information possible ;
- une **output gate** \(o_t\), qui contrôle ce qui est transmis à l’état caché.

Les équations standard sont :

\[
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
\]

\[
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
\]

\[
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
\]

\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]

\[
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
\]

\[
h_t = o_t \odot \tanh(c_t)
\]

Ici, \(\sigma\) est la fonction sigmoïde et \(\odot\) désigne le produit terme à terme. Conceptuellement, cela veut dire que le modèle apprend à distinguer :

- ce qui doit être **retenu** d’un régime passé,
- ce qui doit être **oublié**,
- et ce qui est **important maintenant** pour prédire le retour futur.

Dans notre cas, c’est particulièrement adapté car les marchés alternent entre des phases de tendance, de volatilité élevée, de consolidation ou de stress. Le LSTM permet d’encoder ce contexte de manière dynamique.

### Construction des séquences et contraintes temporelles

Un point absolument essentiel dans un projet de finance est l’absence de **fuite d’information (data leakage)**. Pour une date \(t\), le modèle ne doit jamais utiliser d’information provenant de \(t+1\) ou du futur. Cela impose plusieurs contraintes.

La première contrainte concerne la construction des features. Toutes les fenêtres mobiles (*rolling windows*) doivent être **alignées sur le passé**, c’est-à-dire construites avec les observations jusqu’à \(t\) uniquement. Par exemple, une volatilité réalisée sur 20 jours est calculée avec les retours de \(t-19\) à \(t\), jamais au-delà.

La deuxième contrainte concerne la cible. Si l’on prédit \(y_t = \log(C_{t+1}/C_t)\), alors la ligne datée \(t\) contient les features disponibles à la fin de la journée \(t\), et la cible correspond au mouvement futur entre \(t\) et \(t+1\).

La troisième contrainte concerne les séparations entre données d’entraînement, de validation et de test. En série temporelle, on ne peut pas faire de *shuffle* aléatoire comme dans un problème classique. Les données doivent être séparées **chronologiquement**. Typiquement :

- **train** : partie ancienne de l’historique ;
- **validation** : période intermédiaire utilisée pour choisir les hyperparamètres et contrôler l’overfitting ;
- **test** : période récente, jamais utilisée pendant l’apprentissage.

Si l’on note \(T_{\text{train}} < T_{\text{val}} < T_{\text{test}}\), alors on impose :

\[
\text{Train} = \{t \leq T_{\text{train}}\}, \quad
\text{Val} = \{T_{\text{train}} < t \leq T_{\text{val}}\}, \quad
\text{Test} = \{T_{\text{val}} < t \leq T_{\text{test}}\}
\]

Cette séparation temporelle est non négociable en finance, car l’objectif est de simuler une situation réaliste : prédire l’avenir avec uniquement le passé disponible.

### Normalisation des données

Avant d’entrer dans le réseau, les features sont généralement **normalisées** (*feature scaling*). Le point critique est que les paramètres de normalisation — moyenne et écart-type — doivent être estimés **uniquement sur l’ensemble d’entraînement**, puis appliqués à la validation et au test. Si l’on calculait ces statistiques sur l’ensemble complet, on réintroduirait une fuite d’information.

Si une feature brute est \(x\), une normalisation standard s’écrit :

\[
x^{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \varepsilon}
\]

où \(\mu_{\text{train}}\) et \(\sigma_{\text{train}}\) sont calculés uniquement sur le train.

### Fonction de coût et apprentissage

L’entraînement consiste à ajuster les paramètres \(\theta\) pour minimiser une fonction de coût. Dans ce type de projet, la perte la plus simple est généralement l’erreur quadratique moyenne (*Mean Squared Error*, **MSE**) :

\[
\mathcal{L}(\theta) = \frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2
\]

Cette fonction pénalise fortement les grosses erreurs. Elle pousse le modèle à apprendre la meilleure approximation moyenne du retour futur.

L’optimisation se fait ensuite par **descente de gradient** avec un algorithme comme **Adam**, qui adapte dynamiquement le pas d’apprentissage (*learning rate*). À chaque époque (*epoch*), le modèle parcourt les données d’entraînement, calcule la perte, puis met à jour ses paramètres.

Comme les données financières sont très bruitées, on utilise généralement aussi :

- un **early stopping**, qui arrête l’apprentissage lorsque la performance sur la validation ne s’améliore plus ;
- éventuellement du **dropout** ou d’autres mécanismes de régularisation pour limiter l’overfitting.

Le nombre de jours dans la fenêtre \(L\), la taille de l’état caché, le nombre de couches, le learning rate ou encore la patience de l’early stopping sont des **hyperparamètres**. Ils influencent fortement les performances et doivent être choisis avec prudence.

---

## Évaluation des performances

Une fois le modèle entraîné, il faut l’évaluer sur des données jamais vues auparavant, c’est-à-dire sur le **jeu de test**. L’évaluation a deux dimensions complémentaires :

1. une dimension **statistique**, qui mesure la qualité de la prédiction ;
2. une dimension **financière**, qui mesure l’utilité réelle du signal pour une stratégie.

L’erreur brute n’est pas suffisante en finance. Un modèle peut avoir une erreur faible tout en étant inutile pour prendre des décisions. Inversement, un modèle peut avoir une erreur absolue importante mais réussir à classer correctement les jours “favorables” et “défavorables”, ce qui est souvent plus utile.

### 1. Métriques statistiques

La première métrique classique est la **MSE** :

\[
\text{MSE} = \frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2
\]

Elle quantifie l’erreur quadratique moyenne entre retour réel et retour prédit.

On utilise aussi la **RMSE** (*Root Mean Squared Error*) :

\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

Elle a l’avantage d’être exprimée dans la même unité que la cible.

Une autre métrique importante est la **MAE** (*Mean Absolute Error*) :

\[
\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|
\]

La MAE est moins sensible aux valeurs extrêmes que la MSE.

### 2. Corrélation entre prédictions et valeurs réelles

En finance, on utilise beaucoup l’**Information Coefficient** (**IC**), qui est la corrélation de Pearson entre \(\hat{y}_t\) et \(y_t\) :

\[
IC = \text{corr}(\hat{y}_t, y_t)
\]

Un IC positif signifie que les prédictions ont tendance à aller dans le bon sens.

On calcule aussi le **Rank IC**, qui est la corrélation de Spearman entre le rang des prédictions et le rang des réalisations. Cette métrique mesure la capacité du modèle à classer correctement les situations.

### 3. Hit ratio

Le **hit ratio** correspond à la proportion de fois où le modèle prédit correctement le signe du retour :

\[
\text{Hit Ratio} = \frac{1}{N}\sum_{t=1}^{N}\mathbf{1}\big(\text{sign}(\hat{y}_t) = \text{sign}(y_t)\big)
\]

Même un hit ratio légèrement supérieur à 50% peut être exploitable en finance.

### 4. Analyse par déciles

Les prédictions sont triées puis regroupées en déciles. On calcule ensuite le retour moyen observé dans chaque groupe. Si le modèle est pertinent, les déciles les plus élevés doivent correspondre aux meilleurs retours réels.

Cette approche permet d’évaluer la capacité du modèle à **ordonner** les opportunités.

### 5. Analyse des résidus

Les résidus sont définis par :

\[
\varepsilon_t = y_t - \hat{y}_t
\]

On analyse leur distribution, leur variance et leur autocorrélation. Si les résidus contiennent encore de la structure, cela signifie que le modèle peut encore être amélioré.

### 6. Importance des features

Deux approches principales sont utilisées :

- **Permutation importance** : on perturbe une feature et on mesure la dégradation de performance ;
- **Feature ablation** : on supprime une feature et on mesure la variation de métriques :

\[
\Delta IC_i = IC_{\text{full}} - IC_{-i}
\]

\[
\Delta RMSE_i = RMSE_{-i} - RMSE_{\text{full}}
\]

Ces méthodes permettent d’identifier les variables réellement utiles.

### 7. Objectif final

L’objectif n’est pas uniquement de prédire correctement, mais de produire un signal exploitable pour une stratégie d’investissement. L’évaluation du modèle doit donc toujours être liée à son utilisation pratique.

En résumé, le pipeline complet consiste à :

1. entraîner le modèle sur le train ;
2. valider les hyperparamètres sur la validation ;
3. évaluer sur le test ;
4. transformer les prédictions en décisions d’investissement.

C’est cette cohérence entre modèle, évaluation et stratégie qui permet d’obtenir des résultats robustes.

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
