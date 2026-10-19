# Compte rendu du TP2 — Détection du sarcasme en arabe

**Dataset :** ArSarcasm-v2  
**Environnement :** Python, Google Colab, scikit-learn, PyTorch, Hugging Face Transformers  
**Problématique :** classification binaire des tweets arabes en *sarcastiques* et *non sarcastiques*  
**Document de référence :** `TP2_Arabic_Sarcasm_FINAL.ipynb`

## 1. Introduction et objectifs

Le sarcasme est une forme de langage figuré dont le sens implicite peut contredire le sens littéral. Sa détection automatique est particulièrement délicate dans les publications arabophones en raison des variations dialectales, de l'ironie implicite, de l'utilisation des emojis et de la brièveté des tweets.

Ce TP compare plusieurs familles de méthodes de traitement automatique du langage naturel (NLP) :

- **Machine Learning classique :** TF-IDF + Linear SVM ;
- **Approches sans apprentissage supervisé spécifique :** Zero-Shot NLI et heuristique fondée sur le sentiment ;
- **Deep Learning avec embeddings statiques :** BiLSTM, AraVec et mécanisme d'attention ;
- **Transformer léger :** Mini-Transformer avec et sans encodage positionnel ;
- **Modèles pré-entraînés (PLM) :** AraBERT gelé, AraBERT et MARBERTv2 fine-tunés ;
- **Multi-Task Learning :** prédiction conjointe du sarcasme et du sentiment.

L'objectif principal est de maximiser le **F1-score de la classe sarcastique**, tout en examinant la généralisation aux tweets jamais vus, les changements de distribution, les dialectes et les caractéristiques superficielles des textes.

## 2. Dataset et protocole expérimental

Le jeu **ArSarcasm-v2** comprend les colonnes `tweet`, `sarcasm`, `sentiment` et `dialect`.

| Partition | Nombre de tweets | Taux de sarcasme |
|---|---:|---:|
| Train | 11 293 | 17,28 % |
| Validation | 1 255 | 17,29 % |
| **Test officiel** | **3 000** | **27,37 %** |
| **Données d'apprentissage initiales** | **12 548** | — |

Les 12 548 tweets fournis pour l'apprentissage ont été divisés en **90 % Train / 10 % Validation**, avec `random_state=42` et **stratification sur `sarcasm`**. Les **3 000 tweets du Test officiel** constituent un jeu séparé. Les seuils sont déterminés sur Validation et ne doivent pas être ajustés sur Test.

### 2.1. Exploration des données (EDA)

L'étude examine la distribution des classes, les dialectes arabes, la présence de liens ou de mentions et certains indices textuels du sarcasme. Les proportions observées montrent un **déséquilibre de classes** dans les données d'entraînement : la classe sarcastique est minoritaire.

### 2.2. Prétraitement

Les opérations explorées dans le notebook incluent :

1. **Nettoyage par expressions régulières (Regex)** : traitement des URL, mentions et marqueurs de retweet.
2. **Normalisation orthographique arabe** : harmonisation de variantes de `ا` et de `ى/ي`, suppression des diacritiques et du tatweel.
3. **Normalisation des espaces**.
4. **Stop Words Removal**, avec conservation de termes de négation importants pour le sens.
5. **Tokenization et Vectorization** selon la famille de modèles : TF-IDF, vocabulaire séquentiel ou tokeniseur pré-entraîné.

Les transformations apprises, comme le vocabulaire TF-IDF, sont ajustées sur Train uniquement, puis appliquées à Validation et Test.

## 3. Méthodes évaluées

### 3.1. Partie A — Baseline TF-IDF + Linear SVM

La baseline combine un **TF-IDF de mots** (`ngram_range=(1, 2)`) et un **TF-IDF de caractères** (`ngram_range=(2, 5)`). Un classifieur **LinearSVC** apprend la frontière de décision. Le seuil de décision retenu après optimisation sur Validation est **−0,4070**.

Cette approche constitue une référence rapide, interprétable à travers ses caractéristiques lexicales, et utilisable sur CPU.

### 3.2. Partie B — Zero-Shot et heuristique de sentiment

Deux approches complémentaires sont examinées :

- **Zero-Shot NLI** avec un modèle multilingue d'inférence en langage naturel ;
- **Sentiment-Based Heuristic** avec CAMeLBERT, en utilisant le sentiment comme indice indirect du sarcasme.

Ces méthodes illustrent les limites d'un transfert direct depuis une autre tâche : une expression négative n'est pas nécessairement sarcastique, et un message sarcastique ne porte pas obligatoirement un sentiment négatif.

### 3.3. Partie C — Embeddings statiques et Deep Learning

Plusieurs variantes de **BiLSTM** sont testées : embeddings aléatoires, embeddings pré-entraînés **AraVec** gelés ou entraînables, puis ajout d'un mécanisme d'**Attention**. Une comparaison est également menée avec un **Mini-Transformer**, avec et sans *Positional Encoding*.

Des essais de traitement du déséquilibre de classes sont présents dans le notebook, notamment la pondération des classes et le rééchantillonnage.

### 3.4. Partie D — Modèles pré-entraînés gelés

Le notebook étudie **AraBERT Frozen** en comparant les représentations **[CLS]** et **Mean Pooling**, associées à un classifieur léger. Une variante utilise un **BiLSTM + Attention** au-dessus des représentations pré-entraînées.

### 3.5. Partie E — Fine-Tuning et Multi-Task Learning

Les expériences comprennent :

- le **fine-tuning mono-tâche d'AraBERT** ;
- le **fine-tuning mono-tâche de MARBERTv2** ;
- l'apprentissage **multi-tâches sarcasme + sentiment**, avec une perte combinée :

\[
\mathcal{L}_{total} = \mathcal{L}_{sarcasm} + \lambda \mathcal{L}_{sentiment}
\]

Dans les expériences documentées, **λ = 0,5** donne le meilleur résultat sur Validation.

## 4. Résultats comparatifs sur Validation

Le tableau ci-dessous présente les **13 configurations principales** reprises dans la synthèse du notebook. Les chiffres correspondent au **F1-score de la classe sarcastique sur Validation**, et non au Test officiel.

| Rang | Modèle | F1 Validation | F1 Test officiel |
|---:|---|---:|---:|
| 1 | **AraBERT Multi-Task (λ = 0,5)** | **0,6788** | Non évalué |
| 2 | AraBERT Fine-Tuned | 0,6601 | Non évalué |
| 3 | MARBERTv2 Fine-Tuned | 0,6578 | Non évalué |
| 4 | AraBERT Frozen + BiLSTM + Attention | 0,6536 | Non évalué |
| 5 | AraBERT Frozen + Mean Pooling | 0,6311 | Non évalué |
| 6 | AraBERT Frozen + CLS | 0,6163 | Non évalué |
| 7 | **Linear SVM + TF-IDF** | **0,5506** | **0,5403** |
| 8 | BiLSTM + Attention + Threshold | 0,5123 | Non évalué |
| 9 | BiLSTM + AraVec Unfrozen | 0,4968 | Non évalué |
| 10 | BiLSTM + AraVec Frozen | 0,4912 | Non évalué |
| 11 | BiLSTM + Random Embeddings | 0,4439 | Non évalué |
| 12 | Mini-Transformer + Positional Encoding | 0,3862 | Non évalué |
| 13 | Mini-Transformer sans Positional Encoding | 0,3852 | Non évalué |

**Résultats complémentaires sur Validation :** Zero-Shot NLI : F1 **0,3217** ; heuristique de sentiment CAMeLBERT : F1 **0,4624**. Ces deux expériences ne figurent pas dans le classement des 13 configurations principales ci-dessus.

**Interprétation :** les modèles pré-entraînés obtiennent les meilleures performances de Validation. Le modèle **AraBERT Multi-Task λ = 0,5** se classe premier, mais il serait incorrect de conclure à sa supériorité sur le **Test officiel** sans l'y évaluer.

## 5. Évaluation sur le Test officiel jamais vu

L'évaluation officielle a été réalisée avec le modèle **Linear SVM + TF-IDF** reconstruit sur CPU. Le F1 sur Validation a été retrouvé à **0,5506**, confirmant la cohérence de la reconstruction. Le seuil **−0,4070**, préalablement choisi sur Validation, a ensuite été appliqué au Test sans réajustement.

| Indicateur | Résultat sur Test |
|---|---:|
| Nombre de tweets | 3 000 |
| Accuracy | 0,7130 |
| Precision (sarcastique) | 0,4810 |
| Recall (sarcastique) | 0,6163 |
| **F1-score (sarcastique)** | **0,5403** |

**Matrice de confusion :**

| Classe réelle / Prédiction | Non sarcastique | Sarcastique |
|---|---:|---:|
| Non sarcastique | **1 633 (TN)** | **546 (FP)** |
| Sarcastique | **315 (FN)** | **506 (TP)** |

La différence entre Validation (**0,5506**) et Test (**0,5403**) est de **−0,0103 point de F1**. La proximité de ces résultats est encourageante pour la baseline, mais ne signifie pas que les distributions sont identiques.

## 6. Tests de généralisation et de robustesse

### 6.1. Prior Shift — Décalage de la fréquence du sarcasme

**Méthode :** comparaison des proportions de tweets sarcastiques dans Train, Validation et Test.

- Train : **17,28 %** ;
- Validation : **17,29 %** ;
- Test : **27,37 %**.

Le Test présente un taux de sarcasme supérieur d'environ **10,09 points de pourcentage** à celui de Train. Cette différence décrit un **changement de prior** (fréquence de la classe positive), susceptible d'influencer notamment la précision et l'interprétation du seuil de décision. Les résultats ne permettent pas, à eux seuls, d'isoler l'effet causal de ce changement sur le F1.

### 6.2. Unseen Dialect — Généralisation au dialecte égyptien

**Méthode : Leave-One-Dialect-Out.** Un SVM est entraîné sans tweets égyptiens, puis évalué sur les mêmes **306 tweets égyptiens du Test officiel** que le SVM entraîné sur tous les dialectes.

| Modèle | Precision | Recall | F1 sur tweets égyptiens |
|---|---:|---:|---:|
| SVM entraîné sur tous les dialectes | 0,6058 | 0,8129 | **0,6942** |
| SVM entraîné sans dialecte égyptien | 0,6111 | 0,6387 | **0,6246** |

Le retrait du dialecte égyptien à l'entraînement entraîne une diminution du **F1 de 0,0696** et du rappel de **0,1742** sur ce sous-ensemble. Cela suggère une dépendance aux exemples dialectaux vus pendant l'apprentissage.

**Limite :** le seuil du modèle général a été conservé pour le modèle excluant l'égyptien. Un protocole plus strict utiliserait une Validation sans égyptien pour choisir séparément son seuil.

### 6.3. Shortcut Analysis — URL et mentions

**Méthode : Subgroup Evaluation.** Les performances du SVM sont calculées selon la présence de liens et de mentions dans les tweets du Test officiel.

| Caractéristique | Présence | Tweets | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| URL | Avec | 440 | 0,0870 | 0,1250 | **0,1026** |
| URL | Sans | 2 560 | 0,4898 | 0,6261 | **0,5496** |
| Mention | Avec | 23 | 0,0000 | 0,0000 | **0,0000** |
| Mention | Sans | 2 977 | 0,4814 | 0,6163 | **0,5406** |

Le F1 est très inférieur dans les tweets contenant des URL. Les statistiques descriptives montrent également que le sarcasme est **beaucoup moins fréquent avec URL** que sans URL, tant dans Train que dans Test ; les différences de F1 doivent donc être interprétées en tenant compte de la composition des groupes.

Le sous-groupe contenant une mention ne comprend que **23 tweets et aucun exemple sarcastique** dans le Test : son F1 égal à zéro n'établit pas une mauvaise détection de sarcasme en présence de mentions. Ces comparaisons ne prouvent pas à elles seules que le modèle exploite les URL ou mentions comme **raccourcis causaux**.

### 6.4. Error Analysis — Faux positifs et faux négatifs

**Méthode : Confusion Matrix Analysis et inspection qualitative d'exemples.**

Le modèle commet **546 faux positifs** et **315 faux négatifs** sur le Test officiel. Le notebook extrait **15 FP** et **15 FN** et prévoit un export CSV contenant les tweets, dialectes, scores et colonnes d'annotation.

Les catégories proposées pour une annotation manuelle sont : contexte nécessaire, ironie implicite, marqueur trompeur, particularité dialectale, ambiguïté d'annotation et autre.

**Important :** l'extraction automatique de 30 exemples est réalisée, mais aucune distribution chiffrée de causes d'erreur n'est revendiquée tant qu'une **annotation qualitative humaine** de ces exemples n'a pas été effectuée.

## 7. Discussion générale

Les résultats de Validation indiquent un avantage des **modèles de langue pré-entraînés** par rapport aux architectures apprises à partir de représentations aléatoires ou statiques. L'apprentissage multi-tâches pourrait contribuer à mieux représenter les relations entre sentiment et sarcasme, mais l'effet observé dépend de la pondération de la perte et des paramètres expérimentaux.

La baseline **Linear SVM + TF-IDF** reste intéressante comme modèle rapide et peu coûteux : elle a pu être reproduite sur CPU et mesurée sur le Test officiel. Ses résultats de robustesse mettent cependant en évidence des écarts importants selon les URL et les dialectes.

La différence de prévalence du sarcasme entre les partitions rend nécessaire une interprétation prudente des métriques. En particulier, il ne faut pas comparer des F1 calculés sur **des partitions différentes** comme s'ils constituaient une comparaison finale équitable.

## 8. Limites du travail

1. **Checkpoints indisponibles :** les poids des modèles BiLSTM et Transformers ont été perdus après une réinitialisation de Google Colab. Les scores de Test correspondants n'ont pas pu être calculés.
2. **Absence de GPU utilisable :** les entraînements lourds n'ont pas été relancés sur CPU, afin d'éviter des temps de calcul excessifs.
3. **Validation ≠ Test :** le classement des 13 configurations correspond aux performances sur Validation uniquement.
4. **Dialecte non vu :** le test réalisé porte sur la baseline SVM et sur le dialecte égyptien ; il ne démontre pas une robustesse globale à tous les dialectes.
5. **Analyse d'erreurs qualitative :** les 30 exemples ont été préparés, mais leur catégorisation doit être vérifiée manuellement.
6. **Robustesse des raccourcis :** les résultats par sous-groupes sont descriptifs ; une ablation contrôlée serait nécessaire pour identifier un mécanisme causal.

## 9. Conclusion

Ce TP a permis de mettre en œuvre un ensemble varié d'approches de **détection du sarcasme en arabe**, allant du Machine Learning classique au **Multi-Task Fine-Tuning de Transformers**. Sur **Validation**, la meilleure configuration est **AraBERT Multi-Task avec λ = 0,5 (F1 = 0,6788)**. Sur le **Test officiel**, le seul modèle disposant d'une évaluation vérifiée est **Linear SVM + TF-IDF (F1 = 0,5403)**.

L'analyse du **Prior Shift**, du **dialecte égyptien non vu**, des **URL/mentions** et des **erreurs de classification** montre que la performance globale ne suffit pas à caractériser la fiabilité d'un système NLP. Les effets de distribution et de variation linguistique doivent être pris en compte lors de toute évaluation.

**Perspective principale :** sauvegarder les checkpoints directement sur un stockage persistant, puis mesurer les performances des meilleurs Transformers sur le Test officiel dans un environnement GPU, en conservant les seuils sélectionnés sur Validation.

---

## 10. Reproductibilité et livrables

- **Notebook :** `TP2_Arabic_Sarcasm_FINAL.ipynb`.
- **Compte rendu :** `compte_rendu.md`.
- **Modèle évaluable sans GPU :** Linear SVM + TF-IDF.
- **Fichiers prévus par le notebook :** tableaux de comparaison des modèles, métriques du Test officiel, `prior_shift.csv`, résultats de dialecte non vu, `shortcut_analysis.csv` et `error_analysis_30_examples.csv`.
- **Bonne pratique :** versionner les scripts et résultats CSV ; stocker les checkpoints volumineux sur Google Drive ou un stockage persistant, plutôt que de supposer qu'ils restent disponibles après la fermeture de Colab.

*Les valeurs de ce compte rendu proviennent des résultats enregistrés dans le notebook et des sorties chiffrées de l'expérience. Aucune performance manquante sur Test n'a été estimée ou inventée.*
