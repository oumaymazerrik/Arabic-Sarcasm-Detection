# Compte rendu — TP2 : Détection du sarcasme en arabe

**Étudiante :** Oumayma Zerrik  
**Formation :** Master Big Data & IoT — ENSAM Mohammedia  
**Dataset :** ArSarcasm-v2  
**Notebook :** `notebooks/TP2_Arabic_Sarcasm_FINAL.ipynb`  
**Environnement :** Python, scikit-learn, PyTorch, Hugging Face, Google Colab ; Docker et uv pour la reproductibilité.

## 1. Introduction

Dans ce TP, j'ai étudié la détection automatique du sarcasme dans les tweets arabes. La principale difficulté est qu'une phrase peut être sarcastique même si les mots semblent positifs ou neutres. Les dialectes, les emojis et le contexte compliquent également la classification.

J'ai comparé une baseline de Machine Learning, des méthodes zero-shot et basées sur le sentiment, des BiLSTM, un Mini-Transformer et des modèles pré-entraînés comme AraBERT et MARBERTv2. La métrique principale est le **F1-score de la classe sarcastique**, plus adaptée que l'accuracy en présence de déséquilibre de classes.

## 2. Données et protocole

ArSarcasm-v2 fournit **12 548 tweets d'entraînement officiels** et **3 000 tweets de test officiels**, avec notamment les colonnes `tweet`, `sarcasm`, `sentiment` et `dialect`. J'ai séparé les données d'apprentissage en **90 % Train / 10 % Validation**, avec `random_state=42` et `stratify=sarcasm`. Le Test officiel est resté séparé pour l'évaluation finale.

| Partition | Tweets | Sarcasme |
|---|---:|---:|
| Train | 11 293 | 17,28 % |
| Validation | 1 255 | 17,29 % |
| Test officiel | 3 000 | 27,37 % |

### Prétraitement

Pour la baseline SVM, j'ai normalisé les caractères arabes (variantes de alif, `ى` vers `ي`), supprimé les diacritiques et le tatweel, normalisé les URL et mentions et retiré les stop words tout en conservant les négations. J'ai utilisé **TF-IDF de mots (1–2-grams) et de caractères (2–5-grams)**. Pour les modèles séquentiels, j'ai préparé des séquences de tokens ; pour AraBERT et MARBERTv2, j'ai utilisé leurs tokenizers. Les vocabulaires et représentations apprises ont été ajustés sur Train, pas sur Test.

## 3. Résultats des expériences

### Comparaison des modèles

| Modèle | F1 Validation | F1 Test officiel |
|---|---:|---:|
| **AraBERT Multi-Task (λ = 0,5)** | **0,6788** | Non évalué |
| AraBERT Fine-Tuned | 0,6601 | Non évalué |
| MARBERTv2 Fine-Tuned | 0,6578 | Non évalué |
| AraBERT Frozen + BiLSTM + Attention | 0,6536 | Non évalué |
| AraBERT Frozen + Mean Pooling | 0,6311 | Non évalué |
| AraBERT Frozen + CLS | 0,6163 | Non évalué |
| **Linear SVM + TF-IDF** | **0,5506** | **0,5403** |
| BiLSTM + Attention + Threshold | 0,5123 | Non évalué |
| BiLSTM + AraVec Unfrozen | 0,4968 | Non évalué |
| BiLSTM + AraVec Frozen | 0,4912 | Non évalué |
| BiLSTM + Random | 0,4439 | Non évalué |
| Mini-Transformer + Positional Encoding | 0,3862 | Non évalué |
| Mini-Transformer sans Positional Encoding | 0,3852 | Non évalué |

Autres approches sur Validation : **Zero-Shot NLI : F1 = 0,3217** ; **heuristique CAMeLBERT : F1 = 0,4624**.

### Test officiel de la baseline SVM

J'ai reconstruit le SVM et retrouvé son F1 de Validation de **0,5506**. Avec le seuil **−0,4070** choisi sur Validation, j'obtiens sur les **3 000 tweets du Test officiel** :

| Accuracy | Precision | Recall | F1 sarcastique |
|---:|---:|---:|---:|
| 0,7130 | 0,4810 | 0,6163 | **0,5403** |

| Réel / Prédit | Non sarcastique | Sarcastique |
|---|---:|---:|
| Non sarcastique | 1 633 (TN) | 546 (FP) |
| Sarcastique | 315 (FN) | 506 (TP) |

Les autres modèles n'ont **pas de F1 Test vérifié** : leurs checkpoints ont été perdus lors d'une réinitialisation de Colab et je n'ai pas relancé les entraînements lourds sans GPU.

## 4. Réponses aux questions du TP

### Q1. Déséquilibre, dialectes et choix de la métrique

J'observe **17,28 %** de tweets sarcastiques dans Train, **17,29 %** dans Validation et **27,37 %** dans le Test officiel. Les taux par dialecte sont calculables par le regroupement `groupby('dialect')` du notebook ; je ne recopie pas de valeurs par dialecte ici, car leur tableau chiffré complet n'est pas disponible dans les sorties conservées.

Un classifieur qui prédit **toujours « non sarcastique »** aurait une accuracy de **82,72 % sur Train**, **82,71 % sur Validation** et **72,63 % sur Test**, mais son **F1 de la classe sarcastique serait 0** partout. Donc l'accuracy seule est trompeuse. Je retiens le **F1 sarcastique** et je regarde aussi la précision et le rappel.

### Q2. Raccourcis liés au format, aux URL et aux mentions

Dans Train, les tweets avec URL ont un taux de sarcasme de **6,58 %**, contre **23,86 %** sans URL. Sur Test, ces taux sont respectivement **3,64 %** et **31,45 %**. La présence d'une URL est donc associée à une fréquence de sarcasme bien plus faible.

Sur le Test officiel, mon SVM obtient :

| Sous-groupe | Tweets | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Avec URL | 440 | 0,0870 | 0,1250 | 0,1026 |
| Sans URL | 2 560 | 0,4898 | 0,6261 | 0,5496 |
| Avec mention | 23 | 0,0000 | 0,0000 | 0,0000 |
| Sans mention | 2 977 | 0,4814 | 0,6163 | 0,5406 |

Le groupe « avec mention » ne comporte **aucun tweet sarcastique**, donc son F1 nul n'est pas interprétable comme une preuve d'échec. Je n'ai pas de score vérifié pour un **classifieur à une seule règle**, distinct de ces évaluations par sous-groupes. Ces différences me montrent qu'il faut vérifier les artefacts du dataset : elles ne prouvent pas que le modèle exploite causalement les URL.

### Q3. Prétraitement et ablation des indices de sarcasme

J'ai adapté le prétraitement aux modèles. Pour SVM, j'ai utilisé normalisation arabe, nettoyage Regex, suppression de stop words avec conservation des négations et TF-IDF. Pour BiLSTM et Mini-Transformer, j'ai utilisé une tokenisation en séquences. Pour les PLM, j'ai gardé la tokenisation adaptée aux modèles pré-entraînés.

Certains nettoyages peuvent enlever des signaux importants : emojis, ponctuation répétée, allongements et rires. Par exemple, retirer les emojis peut faire disparaître un indice d'ironie. **Je n'ai pas de baisse de F1 mesurée après ablation contrôlée**, donc je ne peux pas donner de chiffres fiables pour cette sous-question.

### Q4. Zero-Shot NLI et formulation de l'hypothèse

J'ai testé un modèle **mDeBERTa multilingue NLI** en zero-shot et obtenu **F1 = 0,3217** sur Validation. Un modèle NLI peut être utilisé sans exemples spécifiques de sarcasme parce qu'il a déjà appris à estimer si une prémisse implique une hypothèse. Ici, la prémisse est le tweet et l'hypothèse décrit son caractère sarcastique. L'attention permet de mettre en relation les tokens du tweet et ceux de l'hypothèse dans le modèle.

Le choix des mots et de la langue de l'hypothèse peut changer les scores. J'ai exploré différentes formulations, mais **je ne dispose pas d'un tableau vérifié arabe / français / anglais** pour quantifier cet effet.

### Q5. Heuristique « sentiment négatif ⇒ sarcastique »

Avec **CAMeLBERT**, j'ai obtenu sur Validation : **Precision = 0,3264**, **Recall = 0,7926** et **F1 = 0,4624** (Accuracy = 0,6813). L'heuristique repère beaucoup de tweets sarcastiques, mais produit aussi de nombreux faux positifs. J'en conclus que le sentiment négatif et le sarcasme sont liés dans certains cas sans être équivalents.

Cette méthode n'est pas garantie exempte de contamination : il faudrait examiner précisément les données utilisées au pré-entraînement ou à l'apprentissage du classifieur de sentiment pour exclure tout recouvrement avec le corpus évalué. Je ne peux pas certifier ce point avec les informations disponibles.

### Q6. Déséquilibre : pondération, sur-échantillonnage et seuil

J'ai comparé trois approches pour le **BiLSTM + Attention** :

| Méthode | F1 Validation |
|---|---:|
| Pondération des classes (*Class Weighting*) | 0,5031 |
| Sur-échantillonnage (*Oversampling*) | 0,4756 |
| Optimisation du seuil (*Threshold Tuning*) | **0,5123** |

L'optimisation du seuil est la meilleure dans cet essai. Le sur-échantillonnage doit être appliqué **après la séparation Train/Validation**, et seulement sur Train, sinon des copies d'observations peuvent se retrouver dans les deux partitions et fausser l'évaluation.

### Q7. BiLSTM, attention et Mini-Transformer

| Configuration | F1 Validation |
|---|---:|
| BiLSTM + embeddings aléatoires | 0,4439 |
| BiLSTM + AraVec Frozen | 0,4912 |
| BiLSTM + AraVec Unfrozen | 0,4968 |
| BiLSTM + Attention + seuil optimisé | **0,5123** |
| Mini-Transformer avec Positional Encoding | 0,3862 |
| Mini-Transformer sans Positional Encoding | 0,3852 |

AraVec améliore le BiLSTM par rapport à des embeddings aléatoires, et l'attention apporte un gain supplémentaire. En revanche, je ne peux pas confirmer que les **poids d'attention** se concentrent effectivement sur les emojis, les rires ou la ponctuation : une inspection visuelle annotée est nécessaire. L'encodage positionnel n'apporte ici qu'un gain de **0,0010 F1**, très faible.

### Q8. PLM gelé vs fine-tuné, AraBERT vs MARBERTv2

Sur Validation, **AraBERT Frozen + CLS = 0,6163** et **Frozen + Mean Pooling = 0,6311**, tandis que **AraBERT Fine-Tuned = 0,6601**. Le fine-tuning gagne donc **0,0290 F1** par rapport au meilleur pooling gelé. **MARBERTv2 Fine-Tuned = 0,6578**, soit **0,0023** de moins qu'AraBERT Fine-Tuned dans mes essais.

Avec un encodeur gelé, l'entraînement porte surtout sur la tête de classification, alors que le fine-tuning met à jour les paramètres du Transformer et demande davantage de mémoire et de calcul. Ma variante **AraBERT Frozen + BiLSTM + Attention** comportait **953 090 paramètres entraînables** pour sa partie ajoutée et a obtenu **0,6536**. Je ne dispose pas de mesures chronométrées homogènes de durée et de latence pour comparer rigoureusement les coûts. Sur Validation, je choisirais AraBERT Fine-Tuned plutôt que MARBERTv2 pour son F1 légèrement supérieur, mais l'écart est trop faible pour conclure sans répétitions et Test officiel.

### Q9. Multi-Task Learning : influence du sentiment et de λ

J'ai utilisé une perte combinée :

`L_total = L_sarcasm + λ × L_sentiment`

| Expérience | F1 Validation |
|---|---:|
| AraBERT mono-tâche | 0,6601 |
| Multi-Task, λ = 0,3 | 0,6517 |
| **Multi-Task, λ = 0,5** | **0,6788** |
| Multi-Task, λ = 1,0 | Résultat final non confirmé |

Avec **λ = 0,5**, le gain sur le modèle mono-tâche est de **0,0187 F1**. Je pense que la tâche sentiment peut aider l'encodeur à apprendre des informations utiles au sarcasme. Mais si son poids est mal choisi, elle peut aussi perturber la tâche principale : l'essai avec λ = 0,3 est inférieur au mono-tâche. L'amélioration reste à confirmer sur Test.

### Q10. Prior Shift et recalibration

Le taux de sarcasme passe de **17,29 % sur Validation** à **27,37 % sur Test**, soit environ **+10,08 points de pourcentage**. J'ai gardé le seuil de décision du SVM fixé sur Validation à **−0,4070**. Son F1 passe de **0,5506** sur Validation à **0,5403** sur Test.

Je n'ai **pas calculé le seuil optimal a posteriori sur Test**, donc je ne peux pas donner numériquement l'écart entre les deux seuils. En production, je suivrais la fréquence de sarcasme prédite, la dérive des données et les performances sur un échantillon régulièrement annoté. Je réajusterais le seuil uniquement sur des données de validation récentes distinctes du test d'évaluation, avec recalibration si nécessaire.

### Q11. Dialecte non vu et déploiement au Maroc

J'ai réalisé une expérience **Leave-One-Dialect-Out** : retirer les tweets égyptiens du Train, réentraîner le SVM puis évaluer sur **306 tweets égyptiens** du Test officiel.

| Entraînement | Precision | Recall | F1 sur Egypt |
|---|---:|---:|---:|
| Tous les dialectes | 0,6058 | 0,8129 | **0,6942** |
| Sans tweets égyptiens | 0,6111 | 0,6387 | **0,6246** |

Le F1 baisse de **0,0696**. La présence de données du dialecte cible pendant l'apprentissage a donc compté dans cette expérience.

Pour le dialecte **`magreb`**, le notebook inclut le comptage par dialecte, mais **je n'ai pas de valeurs numériques vérifiées de Train et Test à reprendre dans ce rapport**. Avant un déploiement sur des tweets marocains, je constituerais un jeu marocain annoté et indépendant, j'évaluerais les métriques par dialecte et je ferais relire les erreurs par des locuteurs concernés. Le test égyptien ne suffit pas pour garantir les performances en darija marocaine.

### Q12. Analyse d'erreurs et choix d'un modèle en production

Sur le Test officiel, le SVM donne **546 faux positifs** et **315 faux négatifs**. J'ai préparé **15 FP et 15 FN** pour une analyse manuelle portant sur l'ironie implicite, le contexte, les marqueurs trompeurs, les dialectes et le bruit d'annotation. Je ne peux pas annoncer de **catégorie dominante** ou de **pourcentage de bruit d'annotation** sans avoir annoté les 30 exemples.

Pour un prototype exécutable localement, je retiendrais **Linear SVM + TF-IDF**, avec le seuil **−0,4070**, parce que c'est le seul modèle dont j'ai vérifié le **F1 Test = 0,5403** et qu'il s'exécute sans GPU. Je ne prétends pas qu'il est le plus performant en production : **AraBERT Multi-Task** a le meilleur score de Validation, mais son F1 Test, sa latence et son coût d'inférence n'ont pas été mesurés. Il faut comparer ces aspects avant de changer de modèle.

## 5. Décision et justification

Pour la version actuelle du projet, je conserve **Linear SVM + TF-IDF** comme **baseline de référence déployable sur CPU**. Sa précision sur Test est de **0,4810**, son rappel de **0,6163** et son F1 de **0,5403**. Cette configuration est simple à sauvegarder et à exécuter via Python dans un environnement Docker avec uv. Je garderais **AraBERT Multi-Task** comme candidat à réévaluer dès que le GPU et les checkpoints sont disponibles.

## 6. Limites et pistes d'amélioration

- Les checkpoints Transformers ont été perdus après la réinitialisation de Colab ; leurs scores sur Test restent **non évalués**.
- Je n'ai pas de mesures complètes du seuil optimal sur Test, des temps d'entraînement ni des latences d'inférence.
- L'étude des URL et mentions repose sur des sous-groupes, pas sur une ablation causale contrôlée.
- Les 30 erreurs extraites doivent encore être classées manuellement pour estimer les catégories dominantes et le bruit d'annotation.
- Avant un usage sur des tweets marocains, il faut un **Test indépendant en darija marocaine**, idéalement annoté par plusieurs personnes.
- Pour fiabiliser les futures expériences, les modèles entraînés seront sauvegardés directement dans un dossier persistant (`models/` ou Drive), avec leurs configurations et graines aléatoires.

## 7. Conclusion

Ce TP m'a permis de comparer des approches NLP très différentes pour la détection du sarcasme arabe. J'ai constaté que les modèles pré-entraînés donnent globalement de meilleurs scores sur Validation, avec **AraBERT Multi-Task (λ = 0,5)** en tête (**F1 = 0,6788**). J'ai aussi pu reproduire et évaluer la baseline **Linear SVM** sur le Test officiel (**F1 = 0,5403**).

Les analyses montrent qu'un bon F1 global ne suffit pas : la répartition des classes change entre les partitions et les résultats varient selon les URL et le dialecte. Je retiens donc l'importance de séparer strictement Train, Validation et Test, de contrôler les raccourcis et de tester les modèles sur les dialectes visés avant un déploiement.

## 8. Reproductibilité et livrables

```text
Arabic-Sarcasm-Detection/
├── README.md
├── compte_rendu.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── compose.yaml
├── notebooks/
│   └── TP2_Arabic_Sarcasm_FINAL.ipynb
├── data/
├── models/
└── results/
```

Le notebook contient le protocole expérimental et les cellules de comparaison. Les modèles et résultats lourds sont conservés hors Git lorsque nécessaire. La configuration **Docker + uv** vise à faciliter la reproduction locale des étapes compatibles CPU ; elle ne remplace pas l'accès au GPU pour le fine-tuning.
