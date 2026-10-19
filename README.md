Arabic Sarcasm Detection — ArSarcasm-v2
Projet NLP réalisé dans le cadre du TP2 — Détection du sarcasme en arabe. L'objectif est de comparer des approches de Machine Learning, Deep Learning et modèles de langue pré-entraînés pour classifier des tweets arabes comme sarcastiques ou non sarcastiques.
> **Résultat principal (Validation) :** AraBERT Multi-Task, λ = 0,5 — **F1 = 0,6788**.  
> **Résultat vérifié sur le Test officiel :** Linear SVM + TF-IDF — **F1 = 0,5403**.
>
> Les F1 de Validation des modèles Transformers **ne sont pas** des scores sur le Test officiel.
Objectifs
Explorer le dataset ArSarcasm-v2 et le déséquilibre entre les classes.
Prétraiter les tweets arabes : normalisation, URL, mentions, stop words et négations.
Comparer TF-IDF + SVM, Zero-Shot NLI, heuristique de sentiment, BiLSTM, AraVec, Mini-Transformer, AraBERT, MARBERTv2 et apprentissage multi-tâches.
Évaluer la généralisation : Test officiel, Prior Shift, Unseen Dialect, Shortcut Analysis et Error Analysis.
Données et protocole
Partition	Tweets	Proportion sarcastique
Train	11 293	17,28 %
Validation	1 255	17,29 %
Test officiel	3 000	27,37 %
Le jeu d'entraînement d'origine contient 12 548 tweets, répartis en 90 % Train / 10 % Validation (`random_state=42`, stratification sur `sarcasm`). Le Test officiel de 3 000 tweets est distinct. Les seuils de classification sont choisis sur Validation, sans ajustement sur Test.
Colonnes principales : `tweet`, `sarcasm`, `sentiment`, `dialect`.
Méthodes
Famille	Approches étudiées
Baseline ML	TF-IDF mots (1–2-grammes) + caractères (2–5-grammes), Linear SVM
Zero-Shot et heuristiques	NLI multilingue ; CAMeLBERT pour le sentiment
Deep Learning	BiLSTM, AraVec Frozen/Unfrozen, attention
Transformer léger	Mini-Transformer avec/sans Positional Encoding
Pré-entraînés gelés	AraBERT CLS, Mean Pooling, BiLSTM + Attention
Fine-Tuning	AraBERT, MARBERTv2
Multi-Task Learning	Prédiction conjointe sarcasme + sentiment
Métrique principale : F1-score de la classe sarcastique.
Résultats comparatifs
Modèle	F1 Validation	F1 Test officiel
AraBERT Multi-Task (λ = 0,5)	0,6788	Non évalué
AraBERT Fine-Tuned	0,6601	Non évalué
MARBERTv2 Fine-Tuned	0,6578	Non évalué
AraBERT Frozen + BiLSTM + Attention	0,6536	Non évalué
AraBERT Frozen + Mean Pooling	0,6311	Non évalué
AraBERT Frozen + CLS	0,6163	Non évalué
Linear SVM + TF-IDF	0,5506	0,5403
BiLSTM + Attention + Threshold	0,5123	Non évalué
BiLSTM + AraVec Unfrozen	0,4968	Non évalué
BiLSTM + AraVec Frozen	0,4912	Non évalué
BiLSTM + Random Embeddings	0,4439	Non évalué
Mini-Transformer + Positional Encoding	0,3862	Non évalué
Mini-Transformer sans Positional Encoding	0,3852	Non évalué
Expériences complémentaires sur Validation : Zero-Shot NLI : 0,3217 ; heuristique de sentiment CAMeLBERT : 0,4624.
Test officiel — Linear SVM
Accuracy	Precision	Recall	F1-score
0,7130	0,4810	0,6163	0,5403
Matrice de confusion : TN = 1 633, FP = 546, FN = 315, TP = 506. Le seuil de décision fixé sur Validation est −0,4070.
Généralisation et robustesse
Prior Shift : 17,28 % de tweets sarcastiques dans Train contre 27,37 % dans Test.
Unseen Dialect (égyptien, 306 tweets de Test) : F1 du SVM entraîné sur tous les dialectes 0,6942, contre 0,6246 après exclusion des tweets égyptiens de l'entraînement.
Shortcut Analysis : F1 0,1026 avec URL contre 0,5496 sans URL. Avec mentions : 23 tweets, sans exemple sarcastique ; le F1 à zéro n'est pas interprétable comme une preuve de dégradation.
Error Analysis : 546 faux positifs et 315 faux négatifs ; extraction de 15 exemples de chaque type pour une annotation qualitative.
Les analyses par sous-groupe sont descriptives : elles ne démontrent pas à elles seules que le modèle utilise des raccourcis de façon causale.
Structure du dépôt
```text
Arabic-Sarcasm-Detection/
├── README.md
├── compte_rendu.md
└── TP2_Arabic_Sarcasm_FINAL.ipynb
```
Les fichiers CSV et graphiques peuvent être produits par les cellules finales du notebook. Ils ne sont pas supposés présents dans le dépôt tant qu'ils n'ont pas été exportés.
Exécution avec Google Colab
Ouvrir `TP2_Arabic_Sarcasm_FINAL.ipynb` dans Google Colab.
Installer les bibliothèques nécessaires dans les cellules prévues par le notebook ; les principaux outils utilisés sont `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `torch`, `transformers`, `gensim` et `emoji`.
Pour une exécution sans GPU, se concentrer sur la reconstruction du Linear SVM et les cellules finales de Test officiel, Prior Shift, Unseen Dialect, Shortcut Analysis et Error Analysis.
Exécuter les blocs Partie G — Bilan final et livrables pour obtenir les tableaux et exports, en respectant les dépendances des cellules précédentes.
Attention : l'exécution intégrale des entraînements BiLSTM/Transformers peut être longue et nécessite idéalement un GPU. Les résultats de Validation historiques ne remplacent pas une nouvelle évaluation sur le Test officiel.
Limites
Les checkpoints des modèles Deep Learning et Transformers n'étaient plus disponibles après la réinitialisation de Colab. En l'absence de GPU utilisable, ils n'ont pas été réentraînés et leurs F1 Test restent non évalués. Les résultats de l'analyse d'erreurs doivent aussi être complétés par l'annotation manuelle des exemples extraits.
Documentation
Consulter `compte_rendu.md` pour le protocole expérimental, les résultats détaillés, l'interprétation des tests de robustesse et les limites.
---
Technologies : Python · Google Colab · NLP · scikit-learn · PyTorch · Hugging Face Transformers · AraBERT · MARBERTv2
