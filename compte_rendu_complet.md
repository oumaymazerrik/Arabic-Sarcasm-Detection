# Compte rendu — TP2 : Détection du sarcasme en arabe

**Étudiante :** Oumayma Zerrik  
**Formation :** Master Big Data & Internet of Things — ENSAM Casablanca  
**Dataset :** ArSarcasm-v2  
**Notebook principal :** `arabic-sarcasm-final-organized.ipynb`  
**Environnement :** Python, scikit-learn, PyTorch, Hugging Face Transformers, Google Colab, Kaggle GPU (Tesla T4), Docker et uv.

---

## 1. Introduction

Dans ce travail pratique, j'ai étudié la **détection automatique du sarcasme dans les tweets arabes** à travers plusieurs approches de Natural Language Processing (NLP), allant du Machine Learning classique aux Transformers pré-entraînés.

Le sarcasme constitue un problème complexe, car le sens implicite d'une phrase peut différer de son sens littéral. Une expression positive peut, par exemple, être utilisée pour transmettre une critique ou une intention ironique.

Cette difficulté est accentuée en arabe par :

- La diversité des dialectes.
- Les variations orthographiques.
- Les expressions familières.
- Les emojis, les rires et la ponctuation.
- La dépendance au contexte.
- Le déséquilibre entre tweets sarcastiques et non sarcastiques.

L'objectif principal consiste à comparer plusieurs architectures, puis à mesurer leur capacité de généralisation sur des données jamais utilisées pendant l'entraînement.

J'ai expérimenté des méthodes TF-IDF, Zero-Shot, des modèles basés sur le sentiment, des BiLSTM, des Mini-Transformers, ainsi que des modèles pré-entraînés tels qu'AraBERT et MARBERTv2.

La métrique principale retenue est le **F1-score de la classe sarcastique**, car cette métrique combine précision et rappel et reste plus informative que l'accuracy seule dans un contexte de classes déséquilibrées.

---

## 2. Données et protocole expérimental

### 2.1 Présentation du dataset

Le dataset **ArSarcasm-v2** contient des tweets arabes annotés selon différentes caractéristiques.

Les principales colonnes utilisées sont :

| Colonne | Description |
|---|---|
| `tweet` | Texte original du tweet |
| `sarcasm` | Étiquette binaire de sarcasme |
| `sentiment` | Sentiment NEG, NEU ou POS |
| `dialect` | Dialecte associé au tweet |

Le corpus comprend **12 548 tweets dans le jeu d'entraînement officiel** et **3 000 tweets dans le Test officiel**.

Le jeu d'entraînement officiel a été divisé avec :

- 90 % pour Train.
- 10 % pour Validation.
- `random_state=42`.
- `stratify=df["sarcasm"]`.

### 2.2 Distribution des données

| Partition | Nombre de tweets | Taux de sarcasme |
|---|---:|---:|
| Train | 11 293 | 17,28 % |
| Validation | 1 255 | 17,29 % |
| Test officiel | 3 000 | 27,37 % |

Le Test officiel est resté séparé de Train et Validation pour l'évaluation finale.

Les taux de sarcasme révèlent un déséquilibre de classes, ainsi qu'une différence de prévalence entre les données de développement et le Test officiel.

### 2.3 Prétraitement des textes

Les méthodes appliquées dépendent de l'architecture.

**Pour Linear SVM :**

- Nettoyage avec expressions régulières.
- Normalisation des caractères arabes.
- Remplacement des variantes de `أ`, `إ` et `آ` par `ا`.
- Transformation de `ى` en `ي`.
- Suppression des diacritiques et du Tatweel.
- Normalisation des URL et mentions.
- Suppression des stop words avec conservation des négations.
- Extraction TF-IDF Word N-grams (1,2).
- Extraction TF-IDF Character N-grams (2,5).

**Pour les modèles Deep Learning :**

- Préparation de séquences de tokens.
- Construction de représentations numériques.
- Padding et troncature.
- Utilisation d'embeddings adaptés aux architectures.

**Pour AraBERT et MARBERTv2 :**

- Tokenizers pré-entraînés correspondants.
- Subword Tokenization.
- `input_ids` et `attention_mask`.
- Longueur maximale de 128 tokens.

Les vocabulaires TF-IDF sont ajustés sur Train uniquement afin d'éviter les fuites d'information vers Validation ou Test.

---

## 3. Comparaison des approches expérimentales

### 3.1 Partie A — Machine Learning classique

J'ai commencé par une baseline utilisant **Linear SVM et TF-IDF**.

J'ai étudié les représentations lexicales basées sur les mots et les caractères, ainsi que leur combinaison.

Le modèle permet une classification relativement rapide avec des ressources CPU limitées.

**Résultats :**

| Métrique | Valeur |
|---|---:|
| F1 Validation | 0,5506 |
| F1 Test officiel | 0,5403 |

**Conclusion :** Linear SVM constitue une baseline utile et reproductible, mais les caractéristiques lexicales restent limitées pour représenter les intentions implicites du sarcasme.

### 3.2 Partie B — Zero-Shot et sentiment

Deux méthodes ont été étudiées.

| Approche | F1 Validation |
|---|---:|
| Zero-Shot NLI multilingue | 0,3217 |
| Heuristique CAMeLBERT | 0,4624 |

L'approche Zero-Shot utilise un modèle NLI pour évaluer si un tweet implique une hypothèse exprimant le sarcasme.

L'heuristique CAMeLBERT assimile certaines prédictions de sentiment négatif à une indication de sarcasme.

**Conclusion :** ces approches présentent un intérêt expérimental, mais elles ne remplacent pas un modèle entraîné spécifiquement sur des exemples annotés de sarcasme. Un sentiment négatif ne correspond pas nécessairement à du sarcasme.

### 3.3 Partie C — Embeddings statiques et Deep Learning

J'ai comparé plusieurs architectures séquentielles.

| Modèle | F1 Validation |
|---|---:|
| BiLSTM + Random Embeddings | 0,4439 |
| BiLSTM + AraVec Frozen | 0,4912 |
| BiLSTM + AraVec Unfrozen | 0,4968 |
| BiLSTM + Attention + Threshold | **0,5123** |
| Mini-Transformer + Positional Encoding | 0,3862 |
| Mini-Transformer sans Positional Encoding | 0,3852 |

Les embeddings AraVec améliorent les performances du BiLSTM par rapport aux embeddings initialisés aléatoirement.

L'utilisation d'Attention et l'optimisation du seuil permettent une amélioration supplémentaire dans les expériences réalisées.

Les Mini-Transformers entraînés à partir de zéro obtiennent des performances plus faibles que les modèles utilisant des représentations linguistiques pré-entraînées.

**Conclusion :** les connaissances acquises au préalable par les embeddings pré-entraînés sont importantes pour un dataset de taille limitée.

### 3.4 Partie D — Pretrained Language Models gelés

J'ai utilisé AraBERT comme extracteur de représentations contextualisées, sans mettre à jour son encodeur.

| Configuration | F1 Validation |
|---|---:|
| AraBERT Frozen + CLS | 0,6163 |
| AraBERT Frozen + Mean Pooling | 0,6311 |
| AraBERT Frozen + BiLSTM + Attention | **0,6536** |

Le Mean Pooling obtient un meilleur résultat que CLS dans cette expérience.

L'ajout de BiLSTM et d'Attention permet d'atteindre un F1 Validation de 0,6536.

**Conclusion :** un modèle de langue pré-entraîné peut fournir des représentations contextualisées performantes même lorsque ses paramètres restent gelés.

### 3.5 Partie E — Fine-Tuning des Transformers

Afin d'améliorer les résultats, j'ai réalisé de nouveaux entraînements sur Kaggle à l'aide d'un GPU NVIDIA Tesla T4.

Les principaux modèles sont :

- AraBERT Fine-Tuned.
- MARBERTv2 Fine-Tuned.
- AraBERT Multi-Task Learning.

Les expériences utilisent notamment Weighted Cross-Entropy pour la tâche sarcasme, un learning rate de `2e-5`, un batch size de 16 et une longueur maximale de 128 tokens.

Le meilleur checkpoint est sélectionné en fonction du F1 Validation, avec Early Stopping pour les entraînements concernés.

#### AraBERT Fine-Tuned

Modèle : `aubmindlab/bert-base-arabertv02-twitter`

| Métrique | Résultat |
|---|---:|
| F1 Validation | 0,6601 |
| Accuracy Test | 0,7667 |
| Precision Test | 0,5565 |
| Recall Test | 0,7259 |
| **F1 Test** | **0,6300** |

#### MARBERTv2 Fine-Tuned

Modèle : `UBC-NLP/MARBERTv2`

| Métrique | Résultat |
|---|---:|
| F1 Validation | 0,6623 |
| Accuracy Test | 0,7800 |
| Precision Test | 0,5855 |
| Recall Test | 0,6711 |
| **F1 Test** | **0,6254** |

#### AraBERT Multi-Task Learning

Le modèle utilise un encodeur AraBERT partagé avec deux têtes de classification :

- Sarcasm Head : classification binaire.
- Sentiment Head : NEG, NEU et POS.

La fonction de perte est :

`L_total = L_sarcasm + 0.5 × L_sentiment`

| Métrique | Résultat |
|---|---:|
| F1 Validation | **0,6730** |
| Accuracy Test | 0,7570 |
| Precision Test | 0,5401 |
| Recall Test | **0,7552** |
| F1 Test | 0,6298 |

Le meilleur checkpoint Multi-Task correspond à la deuxième époque.

L'Early Stopping s'est déclenché après deux époques supplémentaires sans amélioration.

**Conclusion :** les Transformers fine-tunés dépassent nettement la baseline classique sur le Test officiel. Le Multi-Task obtient le meilleur F1 Validation, mais l'amélioration ne se traduit pas par un gain clair de F1 Test par rapport à AraBERT Mono-Task.

---

## 4. Tableau comparatif final

| Modèle | F1 Validation | F1 Test officiel |
|---|---:|---:|
| **AraBERT Multi-Task (λ = 0,5)** | **0,6730** | 0,6298 |
| MARBERTv2 Fine-Tuned | 0,6623 | 0,6254 |
| AraBERT Fine-Tuned | 0,6601 | **0,6300** |
| AraBERT Frozen + BiLSTM + Attention | 0,6536 | Non évalué |
| AraBERT Frozen + Mean Pooling | 0,6311 | Non évalué |
| AraBERT Frozen + CLS | 0,6163 | Non évalué |
| Linear SVM + TF-IDF | 0,5506 | 0,5403 |
| BiLSTM + Attention + Threshold | 0,5123 | Non évalué |
| BiLSTM + AraVec Unfrozen | 0,4968 | Non évalué |
| BiLSTM + AraVec Frozen | 0,4912 | Non évalué |
| BiLSTM + Random | 0,4439 | Non évalué |
| Mini-Transformer + Positional Encoding | 0,3862 | Non évalué |
| Mini-Transformer sans Positional Encoding | 0,3852 | Non évalué |

Le meilleur F1 Validation est obtenu par AraBERT Multi-Task.

Le meilleur F1 Test observé est obtenu par AraBERT Fine-Tuned.

Leurs F1 Test sont presque identiques, ce qui ne permet pas de conclure à une différence statistiquement significative sans analyse complémentaire.

---

## 5. Évaluation sur le Test officiel

### 5.1 Protocole

Le Test officiel comprend **3 000 tweets jamais utilisés pendant les phases d'entraînement et de sélection sur Validation**.

Pour les Transformers, les prédictions ont été générées en rechargeant les meilleurs checkpoints.

Les résultats des modèles principaux sont les suivants :

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Linear SVM | 0,7130 | 0,4810 | 0,6163 | 0,5403 |
| AraBERT Fine-Tuned | 0,7667 | 0,5565 | 0,7259 | **0,6300** |
| MARBERTv2 Fine-Tuned | **0,7800** | **0,5855** | 0,6711 | 0,6254 |
| AraBERT Multi-Task | 0,7570 | 0,5401 | **0,7552** | 0,6298 |

### 5.2 Matrices de confusion

#### Linear SVM

| Réel / Prédit | Non sarcastique | Sarcastique |
|---|---:|---:|
| Non sarcastique | 1 633 TN | 546 FP |
| Sarcastique | 315 FN | 506 TP |

#### MARBERTv2 Fine-Tuned

| Réel / Prédit | Non sarcastique | Sarcastique |
|---|---:|---:|
| Non sarcastique | 1 789 TN | 390 FP |
| Sarcastique | 270 FN | 551 TP |

#### AraBERT Multi-Task

| Réel / Prédit | Non sarcastique | Sarcastique |
|---|---:|---:|
| Non sarcastique | 1 651 TN | 528 FP |
| Sarcastique | 201 FN | 620 TP |

### 5.3 Interprétation

MARBERTv2 obtient moins de faux positifs que Multi-Task, ce qui explique notamment sa précision plus élevée.

En revanche, AraBERT Multi-Task obtient moins de faux négatifs et détecte davantage de tweets sarcastiques, ce qui lui donne le rappel le plus élevé parmi les modèles évalués.

Il existe donc un compromis entre Precision et Recall.

Le choix d'un modèle dépend des erreurs que l'application cherche prioritairement à réduire.

---

## 6. Réponses aux questions du TP

### Q1. Déséquilibre des classes, dialectes et choix de la métrique

Le dataset présente un déséquilibre significatif :

- Train : 17,28 % de sarcasme.
- Validation : 17,29 %.
- Test officiel : 27,37 %.

Un modèle prédisant systématiquement « non sarcastique » pourrait obtenir une accuracy relativement élevée, mais son F1 de la classe sarcastique serait nul.

C'est pourquoi j'ai utilisé le F1-score comme métrique principale.

La Precision et le Recall permettent également de mieux comprendre les types d'erreurs.

La diversité des dialectes nécessite une évaluation supplémentaire par groupe linguistique.

### Q2. Raccourcis liés aux URL et aux mentions

Certaines caractéristiques du format des tweets peuvent être corrélées avec les étiquettes.

Dans l'analyse historique de Linear SVM :

| Sous-groupe | Tweets | F1 Test |
|---|---:|---:|
| Avec URL | 440 | 0,1026 |
| Sans URL | 2 560 | 0,5496 |
| Avec mention | 23 | 0,0000 |
| Sans mention | 2 977 | 0,5406 |

L'écart de performance entre les tweets avec URL et sans URL est important.

Cependant, cette comparaison ne démontre pas que le modèle utilise causalement la présence d'une URL pour prédire le sarcasme.

Le groupe contenant des mentions est très petit et ne comprend pas de tweet sarcastique dans l'évaluation rapportée. Son F1 nul ne doit donc pas être interprété isolément comme un échec du modèle.

### Q3. Prétraitement et ablation des indices linguistiques

Le prétraitement doit préserver autant que possible les indices utiles à la classification.

Certains éléments peuvent transmettre une intention ironique :

- Emojis.
- Rires.
- Ponctuation répétée.
- Allongements de lettres.
- Expressions dialectales.

Une suppression systématique de ces éléments peut entraîner une perte d'information.

Le notebook comporte une analyse descriptive de plusieurs indices de sarcasme.

En revanche, une ablation contrôlée mesurant la différence de F1 avant et après suppression de chaque indice n'a pas encore été finalisée. Je ne peux donc pas attribuer une baisse de performance chiffrée à ces suppressions.

### Q4. Zero-Shot NLI et formulation des hypothèses

Le Zero-Shot Learning permet de réaliser une classification sans fine-tuning spécifique sur le dataset cible.

J'ai utilisé un modèle multilingue basé sur Natural Language Inference.

Le tweet représente la prémisse, et une formulation décrivant le sarcasme représente l'hypothèse.

Le modèle estime la relation entre ces deux textes.

Le résultat obtenu sur Validation est :

**F1 = 0,3217**

La formulation et la langue des hypothèses peuvent influencer les prédictions. Cependant, je ne dispose pas d'une comparaison quantitative complète et vérifiée pour toutes les formulations testées.

### Q5. Heuristique de sentiment négatif

L'heuristique repose sur l'idée qu'un sentiment négatif peut parfois accompagner une expression sarcastique.

Avec CAMeLBERT, j'ai obtenu :

| Métrique | Validation |
|---|---:|
| Accuracy | 0,6813 |
| Precision | 0,3264 |
| Recall | 0,7926 |
| F1 | 0,4624 |

Le Recall élevé indique que l'heuristique détecte une partie importante des tweets sarcastiques.

Cependant, sa faible précision révèle de nombreux faux positifs.

**Conclusion :** le sentiment négatif ne constitue pas un indicateur suffisant du sarcasme.

Les données de pré-entraînement et d'entraînement du modèle de sentiment devraient être examinées pour écarter un éventuel recouvrement avec le corpus évalué.

### Q6. Traitement du déséquilibre des classes

Pour BiLSTM + Attention, j'ai comparé plusieurs méthodes :

| Méthode | F1 Validation |
|---|---:|
| Class Weighting | 0,5031 |
| Oversampling | 0,4756 |
| Threshold Tuning | **0,5123** |

Le Threshold Tuning obtient le meilleur résultat dans cet essai.

La pondération des classes permet d'augmenter l'importance des erreurs concernant la classe minoritaire.

L'Oversampling augmente le nombre d'observations de la classe minoritaire dans Train.

Il doit impérativement être effectué après la séparation Train/Validation pour éviter une fuite de données.

Le Threshold Tuning ajuste la règle de décision à partir des prédictions obtenues sur Validation.

### Q7. BiLSTM, Attention et Positional Encoding

Les modèles utilisant AraVec dépassent le BiLSTM avec embeddings aléatoires.

L'Attention permet au modèle de construire une représentation pondérée des différentes positions d'une séquence.

Cependant, une amélioration du score ne suffit pas à démontrer que les poids d'Attention correspondent à des indices linguistiques interprétables.

Une visualisation accompagnée d'une annotation manuelle serait nécessaire pour analyser ce phénomène.

Pour le Mini-Transformer :

- Avec Positional Encoding : F1 = 0,3862.
- Sans Positional Encoding : F1 = 0,3852.

La différence observée est de 0,0010.

Le Positional Encoding fournit au Transformer une information sur la position des tokens, mais son effet mesuré reste très faible dans cette expérience.

### Q8. PLM Frozen vs Fine-Tuned et comparaison AraBERT/MARBERTv2

Les modèles Frozen utilisent l'encodeur pré-entraîné comme extracteur de représentations.

Le Fine-Tuning met également à jour les poids du Transformer.

| Modèle | F1 Validation |
|---|---:|
| AraBERT Frozen + CLS | 0,6163 |
| AraBERT Frozen + Mean Pooling | 0,6311 |
| AraBERT Frozen + BiLSTM + Attention | 0,6536 |
| AraBERT Fine-Tuned | 0,6601 |
| MARBERTv2 Fine-Tuned | 0,6623 |

Sur Validation, MARBERTv2 dépasse légèrement AraBERT Fine-Tuned.

En revanche, sur le Test officiel :

- AraBERT : F1 = 0,6300.
- MARBERTv2 : F1 = 0,6254.

AraBERT obtient donc un F1 Test légèrement supérieur.

Le Fine-Tuning demande davantage de ressources qu'une simple extraction d'embeddings Frozen, mais il permet d'adapter les représentations linguistiques à la tâche cible.

Je ne dispose pas de mesures de latence homogènes permettant une comparaison rigoureuse du coût d'inférence de tous les modèles.

### Q9. Multi-Task Learning et influence de λ

L'approche Multi-Task consiste à apprendre plusieurs tâches à l'aide d'un encodeur partagé.

Dans mon cas :

- Tâche principale : détection du sarcasme.
- Tâche auxiliaire : classification du sentiment.

La fonction de perte est :

`L_total = L_sarcasm + λ × L_sentiment`

Les expériences historiques ont donné :

| Configuration | F1 Validation |
|---|---:|
| AraBERT Mono-Task | 0,6601 |
| Multi-Task λ = 0,3 | 0,6517 |
| Multi-Task λ = 0,5 — ancien essai | 0,6788 |
| Multi-Task λ = 1,0 | Résultat final non confirmé |

Dans la nouvelle expérience Kaggle avec λ = 0,5 :

| Métrique | Résultat |
|---|---:|
| F1 Validation | **0,6730** |
| F1 Test officiel | **0,6298** |

Le gain de Validation par rapport à AraBERT Mono-Task est de 0,0129.

En revanche, le F1 Test de Multi-Task reste pratiquement identique à celui d'AraBERT Mono-Task (0,6300).

**Conclusion :** l'apprentissage auxiliaire du sentiment peut améliorer la Validation, mais son avantage en généralisation n'est pas établi par cette expérience.

### Q10. Prior Shift et recalibration

Le taux de sarcasme passe de :

- 17,29 % sur Validation.
- 27,37 % sur Test.

L'écart est d'environ **10,08 points de pourcentage**.

Cette différence représente un changement de la prévalence de la classe sarcastique entre les partitions.

Pour Linear SVM, le seuil sélectionné sur Validation est :

`threshold = -0.4070`

Les scores sont :

- F1 Validation : 0,5506.
- F1 Test : 0,5403.

La légère différence de F1 ne suffit pas à démontrer l'absence de changement de distribution.

Dans une application réelle, je suivrais l'évolution des données entrantes, les proportions de classes, les erreurs et les performances sur des observations annotées.

Toute recalibration devrait être réalisée sur des données adaptées et distinctes du Test officiel utilisé pour l'évaluation.

### Q11. Dialecte non vu et déploiement au Maroc

Une expérience Leave-One-Dialect-Out a été réalisée historiquement avec le dialecte égyptien.

Le protocole consiste à retirer les tweets égyptiens de l'entraînement, à réentraîner Linear SVM sur les autres dialectes et à comparer les prédictions sur les tweets égyptiens du Test officiel.

Les résultats précédemment obtenus sur 306 tweets égyptiens sont :

| Configuration | Precision | Recall | F1 |
|---|---:|---:|---:|
| Tous les dialectes | 0,6058 | 0,8129 | **0,6942** |
| Sans dialecte égyptien | 0,6111 | 0,6387 | **0,6246** |

Le retrait du dialecte cible entraîne une baisse observée de 0,0696 F1.

Ces résultats proviennent de l'expérience précédente ; ils devront être retrouvés dans les sorties de la version finale du notebook pour assurer leur traçabilité complète.

Pour un déploiement au Maroc, il serait indispensable d'évaluer les modèles sur un corpus indépendant de tweets en darija marocaine.

Les performances obtenues sur l'arabe égyptien ne garantissent pas des performances équivalentes sur les dialectes marocains.

### Q12. Analyse d'erreurs et choix du modèle

La baseline Linear SVM produit sur Test :

- **546 False Positives**.
- **315 False Negatives**.

J'ai extrait des exemples de ces deux catégories pour analyser :

- Le sarcasme implicite.
- Les ambiguïtés contextuelles.
- Les expressions dialectales.
- Les indices lexicaux trompeurs.
- Les éventuelles incohérences d'annotation.

Les 30 exemples prévus pour l'analyse qualitative (15 FP et 15 FN) doivent encore être classés manuellement pour identifier les catégories dominantes.

Pour un système de détection du sarcasme, plusieurs choix sont possibles :

**Linear SVM :** adapté comme baseline légère et exécutable sur CPU.

**AraBERT Fine-Tuned :** candidat intéressant lorsque l'objectif principal est de maximiser le F1 observé sur Test.

**MARBERTv2 :** candidat intéressant pour un compromis privilégiant une meilleure précision et moins de faux positifs dans les expériences réalisées.

**AraBERT Multi-Task :** candidat intéressant lorsque le rappel est prioritaire, puisqu'il détecte davantage de tweets sarcastiques.

Le choix final doit également considérer la latence, la mémoire, les ressources d'inférence et les contraintes de déploiement.

---

## 7. Décision et justification

Les nouvelles évaluations changent la conclusion initiale du TP.

Au départ, Linear SVM était le seul modèle avec un F1 Test vérifié, en raison de la perte des anciens checkpoints Transformers après une réinitialisation de Colab.

Les nouveaux entraînements sur Kaggle ont permis de sauvegarder les checkpoints et d'évaluer trois Transformers sur le Test officiel.

Les résultats montrent :

- Linear SVM : F1 Test = 0,5403.
- AraBERT Fine-Tuned : F1 Test = 0,6300.
- MARBERTv2 Fine-Tuned : F1 Test = 0,6254.
- AraBERT Multi-Task : F1 Test = 0,6298.

**Je retiens AraBERT Fine-Tuned comme meilleur modèle selon le F1 Test observé**, tout en soulignant que sa différence avec AraBERT Multi-Task est extrêmement faible.

Linear SVM reste une baseline CPU intéressante.

AraBERT Multi-Task constitue une alternative pertinente pour les applications privilégiant le rappel.

Cette sélection est fondée sur les expériences observées, et non sur une preuve de supériorité statistique ou de performance en production.

---

## 8. Limites et pistes d'amélioration

Les principales limites du travail sont :

1. Les expériences reposent principalement sur une seule séparation Train/Validation.
2. Les entraînements n'ont pas été systématiquement répétés avec plusieurs graines aléatoires.
3. Les modèles Deep Learning historiques n'ont pas tous été évalués sur le Test officiel.
4. Les tests de robustesse concernent principalement Linear SVM.
5. Les différences entre modèles n'ont pas fait l'objet de tests statistiques de significativité.
6. L'analyse qualitative des erreurs n'est pas complètement annotée.
7. Les performances spécifiques sur la darija marocaine doivent encore être mesurées.
8. Le temps d'inférence, l'utilisation mémoire et les coûts de déploiement restent à comparer.

### Améliorations possibles

- Répéter les entraînements sur plusieurs seeds.
- Calculer des intervalles de confiance par Bootstrap.
- Réaliser un McNemar's Test pour comparer les erreurs de deux modèles sur les mêmes exemples.
- Évaluer les Transformers en Leave-One-Dialect-Out.
- Compléter une ablation contrôlée des emojis, rires et ponctuations.
- Étudier d'autres valeurs de λ pour le Multi-Task.
- Annoter qualitativement les faux positifs et faux négatifs.
- Construire un jeu de Test spécifique à l'arabe marocain.

---

## 9. Conclusion générale

Ce travail m'a permis d'explorer plusieurs générations de modèles NLP pour la détection du sarcasme arabe, depuis les méthodes classiques jusqu'aux Transformers pré-entraînés.

Les expériences mettent en évidence trois observations principales.

Premièrement, **les modèles pré-entraînés fine-tunés dépassent nettement la baseline TF-IDF + Linear SVM** sur le Test officiel.

Deuxièmement, **AraBERT Multi-Task atteint le meilleur F1 Validation (0,6730)**, tandis qu'**AraBERT Fine-Tuned obtient le meilleur F1 Test observé (0,6300)**. Cela rappelle qu'un modèle performant sur Validation n'est pas nécessairement supérieur sur des données jamais vues.

Troisièmement, les analyses de robustesse montrent l'importance de considérer la distribution des classes, les variations dialectales, les caractéristiques linguistiques et les erreurs de classification.

L'étude souligne ainsi que l'évaluation d'un modèle NLP ne doit pas se limiter à une seule métrique globale.

Pour une utilisation réelle, il est nécessaire de combiner performance prédictive, robustesse, généralisation linguistique et contraintes de déploiement.

---

## 10. Reproductibilité et livrables

Le projet est disponible dans le repository GitHub :

[Arabic-Sarcasm-Detection](https://github.com/oumaymazerrik/Arabic-Sarcasm-Detection)

### Structure principale

```text
Arabic-Sarcasm-Detection/
├── README.md
├── compte_rendu.md
├── TP2_Arabic_Sarcasm.ipynb
├── TP2_Arabic_Sarcasm_FINAL.ipynb
├── arabic-sarcasm-final-organized.ipynb
└── ... autres fichiers du projet
```

Le notebook organisé regroupe les parties expérimentales, les codes, certaines sorties d'entraînement, les résultats officiels et les analyses de robustesse.

### Gestion des checkpoints

Les meilleurs modèles entraînés ont été archivés :

- `arabert_finetuned_checkpoint.zip`
- `marbertv2_finetuned_checkpoint.zip`
- `arabert_multitask_checkpoint.zip`

Ces archives sont conservées séparément du repository GitHub en raison de leur taille.

Elles permettent de restaurer les modèles sans recommencer les entraînements, sous réserve d'utiliser les architectures et tokenizers correspondants.

### Environnements

- **Google Colab :** premières expériences NLP et Deep Learning.
- **Kaggle GPU Tesla T4 :** Fine-Tuning et évaluation finale des Transformers.
- **Docker et uv :** configuration locale du projet et reproductibilité des tâches compatibles avec l'environnement disponible.

L'utilisation d'un GPU reste recommandée pour reproduire les entraînements complets des Transformers.

---

## 11. Bilan personnel

Ce TP m'a permis de consolider mes connaissances sur :

- Le prétraitement des textes arabes.
- Le Machine Learning appliqué au NLP.
- Les embeddings statiques et contextualisés.
- Les architectures BiLSTM et Transformers.
- Le Fine-Tuning des modèles pré-entraînés.
- Le Multi-Task Learning.
- Les stratégies de gestion du déséquilibre.
- L'évaluation sur données jamais vues.
- Le Prior Shift et la généralisation inter-dialectes.
- L'analyse des erreurs et l'interprétation des métriques.

Il m'a également permis de mieux comprendre l'importance de la reproductibilité expérimentale, de la conservation des checkpoints et de la distinction entre les performances obtenues sur Validation et celles mesurées sur Test.

**Résultat principal : AraBERT Fine-Tuned — F1 Test officiel = 0,6300.**
