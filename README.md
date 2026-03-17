# 📈 Prédiction de séries temporelles financières avec LSTM

## 🧠 Objectif du projet

Ce projet a pour objectif de prédire l’évolution à court terme du prix d’un actif financier (par exemple une action comme Apple) à l’aide d’un modèle de type LSTM (Long Short-Term Memory). Plus précisément, on cherche à estimer le **rendement du lendemain (retour à J+1)**, c’est-à-dire la variation du prix entre aujourd’hui et le jour suivant.

À partir de cette prédiction, l’idée est ensuite de construire une **stratégie d’investissement** : autrement dit, décider quand investir, quand réduire son exposition, et comment gérer le risque.

Ce projet s’inscrit donc à l’intersection de deux domaines :
- le **machine learning**, pour faire des prédictions
- la **finance quantitative**, pour transformer ces prédictions en décisions d’investissement

---

## 🔍 Comprendre le problème

Les marchés financiers évoluent dans le temps, et leurs dynamiques sont souvent complexes, bruitées et difficiles à modéliser. On parle de **série temporelle (time series)**, c’est-à-dire une suite de données ordonnées dans le temps.

Plutôt que d’essayer de prédire directement le prix, ce projet se concentre sur le **rendement (return)**, qui correspond au pourcentage de variation du prix entre deux instants. Ce choix permet de travailler avec des données plus stables et comparables.

Le défi est donc le suivant :

> À partir des informations passées, peut-on prédire le rendement du lendemain ?

---

## 🤖 Pourquoi un LSTM ?

Le modèle utilisé ici est un **LSTM (Long Short-Term Memory)**, un type particulier de réseau de neurones conçu pour traiter des séquences.

Contrairement à des modèles classiques, un LSTM est capable de :
- prendre en compte le **contexte temporel**
- mémoriser des informations importantes dans le passé
- oublier les informations non pertinentes

En d’autres termes, il est particulièrement adapté aux données où **l’ordre dans le temps est essentiel**, comme les séries financières.

---

## 🏗️ Construction des données

Avant d’entraîner le modèle, il est nécessaire de transformer les données brutes en variables exploitables. Cette étape s’appelle le **feature engineering**.

À partir des prix historiques, plusieurs types d’indicateurs sont construits :

- des indicateurs de tendance (ex : moyennes mobiles)
- des indicateurs de volatilité (mesure des variations du marché)
- des indicateurs de momentum (vitesse des mouvements de prix)

Ces variables, appelées **features**, permettent de donner au modèle une vision plus riche du marché que le simple prix.

Les données sont ensuite normalisées (mise à l’échelle) afin de faciliter l’apprentissage du modèle.

---

## 🧪 Entraînement du modèle

Le modèle LSTM est entraîné sur des séquences de données passées pour apprendre à prédire le rendement futur.

Concrètement :
- on fournit au modèle une fenêtre de données (par exemple les 30 derniers jours)
- le modèle apprend à prédire le rendement du jour suivant

Le processus d’entraînement consiste à ajuster les paramètres du modèle pour minimiser l’erreur entre les prédictions et les valeurs réelles.

---

## 📊 Évaluation des performances

Une fois le modèle entraîné, il est évalué sur des données qu’il n’a jamais vues (données de test). Cela permet de vérifier sa capacité à généraliser.

Plusieurs métriques sont utilisées :

- des métriques statistiques (erreur de prédiction)
- des métriques financières (qualité du signal pour le trading)

L’objectif n’est pas uniquement d’avoir une bonne précision, mais surtout un signal **utile pour prendre des décisions d’investissement**.

---

## 💡 De la prédiction à la stratégie

Le modèle produit un signal : une estimation du rendement futur.

Ce signal est ensuite utilisé pour construire une stratégie simple :
- si le rendement prédit est positif → on investit
- s’il est négatif → on réduit ou coupe l’exposition

Ce mécanisme permet de transformer une prédiction en **règle de décision**.

Ensuite, des mécanismes de gestion du risque peuvent être ajoutés pour améliorer la stabilité de la stratégie, notamment en tenant compte de la volatilité du marché.

---

## ⚙️ Organisation du projet

Le projet est structuré en plusieurs étapes :

1. **Collecte et préparation des données**
2. **Construction des features**
3. **Entraînement du modèle LSTM**
4. **Génération des prédictions**
5. **Backtest de stratégies d’investissement**

Chaque étape est pensée pour être reproductible et modulaire, afin de pouvoir tester facilement de nouvelles idées.

---

## 🚀 Conclusion

Ce projet montre comment un modèle de machine learning, en particulier un LSTM, peut être utilisé pour analyser des séries temporelles financières et générer des signaux exploitables.

Il met en évidence un point essentiel :

> La valeur ne vient pas uniquement de la prédiction, mais de la manière dont elle est utilisée pour prendre des décisions.

C’est cette combinaison entre **modélisation et stratégie** qui permet de construire des approches robustes et performantes.

---
