# Arabic Sarcasm Detection — NLP & Deep Learning

## Project Overview

This project investigates **Arabic sarcasm detection** using traditional Machine Learning, Deep Learning, and Pre-trained Language Models (PLMs).

The main objective is to compare different Natural Language Processing approaches for classifying Arabic tweets as sarcastic or non-sarcastic, and to evaluate their generalization capabilities across dialects and linguistic characteristics.

The project includes **13 experimental model configurations**, ranging from TF-IDF-based classifiers to fine-tuned Transformers and Multi-Task Learning.

**Dataset:** ArSarcasm-v2  
**Task:** Binary Classification — Sarcastic vs. Non-Sarcastic Tweets  
**Primary Metric:** F1-score (sarcastic class)  
**Environment:** Python, Google Colab, Kaggle GPU (Tesla T4), PyTorch, Scikit-learn, Hugging Face Transformers

---

## 1. Dataset

The project uses **ArSarcasm-v2**, an Arabic Twitter dataset containing sarcasm, sentiment, and dialect annotations.

Dataset source: [ArSarcasm-v2 GitHub Repository](https://github.com/iabufarha/ArSarcasm-v2)

| Dataset | Number of Tweets |
|---|---:|
| Official Training Set | 12,548 |
| Training Split (90%) | 11,293 |
| Validation Split (10%) | 1,255 |
| Official Test Set | 3,000 |

The official training data is divided using a stratified split with `random_state=42`.

The official test set is kept separate from model training and validation-based model selection.

### Class Distribution

| Dataset | Sarcastic Tweets (%) |
|---|---:|
| Training | 17.28 |
| Validation | 17.29 |
| Official Test | 27.37 |

The official test distribution contains a higher proportion of sarcastic tweets than the development data.

This motivates further investigation of **prior shift and model generalization**.

---

## 2. Arabic Text Preprocessing

The preprocessing pipeline includes:

- URL normalization
- User mention normalization
- Arabic character normalization
- Diacritics and Tatweel removal
- Whitespace normalization
- Removal of retweet markers
- Stop-word removal while preserving negation words where applicable
- Word-level and character-level tokenization
- Transformer subword tokenization

Preprocessing strategies vary according to the selected model.

For Transformer fine-tuning, tokenization is performed using the corresponding pretrained model tokenizer, with a maximum sequence length of 128 tokens.

---

## 3. Experimental Approaches

### Part A — Traditional Machine Learning

The baseline uses **Linear SVM** with:

- Word-Level TF-IDF
- Character-Level TF-IDF
- Combined Word and Character N-grams
- Class imbalance handling
- Stop-word experiments
- Decision threshold optimization

The baseline provides a reproducible reference for evaluating more complex architectures.

### Part B — Zero-Shot and Sentiment-Based Classification

Two approaches were investigated:

- Multilingual NLI Zero-Shot Classification using mDeBERTa
- CAMeLBERT Sentiment-Based Heuristic

These experiments explore the ability of pretrained models to identify sarcasm without direct task-specific fine-tuning.

### Part C — Static Embeddings and Deep Learning

Implemented configurations include:

- BiLSTM with Random Embeddings
- BiLSTM with Frozen AraVec Embeddings
- BiLSTM with Trainable AraVec Embeddings
- BiLSTM with Attention
- Class Weighting and Oversampling
- Threshold Tuning
- Mini-Transformer with Positional Encoding
- Mini-Transformer without Positional Encoding

These experiments examine the contribution of static embeddings, sequence modeling, attention, and class imbalance techniques.

### Part D — Frozen Pre-trained Language Models

AraBERT was used as a frozen feature extractor with:

- CLS Token Representation + Logistic Regression
- Mean Pooling + Logistic Regression
- Frozen AraBERT + BiLSTM + Attention

This part investigates whether contextualized representations improve performance without full Transformer fine-tuning.

### Part E — Transformer Fine-Tuning

Three main models were trained and evaluated:

**1. AraBERT Fine-Tuned**

- Pretrained model: `aubmindlab/bert-base-arabertv02-twitter`
- Binary sarcasm classification
- Weighted Cross-Entropy
- Learning rate: `2e-5`
- Maximum sequence length: `128`
- Batch size: `16`

**2. MARBERTv2 Fine-Tuned**

- Pretrained model: `UBC-NLP/MARBERTv2`
- Binary sarcasm classification
- Weighted Cross-Entropy
- Learning rate: `2e-5`
- Maximum sequence length: `128`
- Batch size: `16`
- Early Stopping

**3. AraBERT Multi-Task Learning**

A shared AraBERT encoder learns two tasks simultaneously:

- Main Task: Sarcasm Detection (2 classes)
- Auxiliary Task: Sentiment Classification (NEG, NEU, POS)

The total training loss is:

`L_total = L_sarcasm + 0.5 × L_sentiment`

The best checkpoint is selected according to sarcasm F1 on the Validation set.

All three Transformer models were trained using GPU acceleration, and their best checkpoints were saved separately.

---

## 4. Experimental Results

### 4.1 Validation Performance

The primary evaluation metric is the F1-score for the sarcastic class.

| Model | F1 Validation |
|---|---:|
| **AraBERT Multi-Task (λ = 0.5)** | **0.6730** |
| MARBERTv2 Fine-Tuned | 0.6623 |
| AraBERT Fine-Tuned | 0.6601 |
| AraBERT Frozen + BiLSTM + Attention | 0.6536 |
| AraBERT Frozen + Mean Pooling | 0.6311 |
| AraBERT Frozen + CLS | 0.6163 |
| Linear SVM + TF-IDF | 0.5506 |
| BiLSTM + Attention + Threshold | 0.5123 |
| BiLSTM + AraVec Unfrozen | 0.4968 |
| BiLSTM + AraVec Frozen | 0.4912 |
| BiLSTM + Random Embeddings | 0.4439 |
| Mini-Transformer + Positional Encoding | 0.3862 |
| Mini-Transformer without Positional Encoding | 0.3852 |

**Best Validation Model: AraBERT Multi-Task (F1 = 0.6730).**

The earlier Multi-Task experiment achieved a validation F1 of 0.6788. The table reports the latest Kaggle training run associated with the saved and officially evaluated checkpoint.

### 4.2 Official Test Performance

The four evaluated models were tested on the **3,000 tweets of the official unseen test set**.

| Model | Accuracy | Precision | Recall | F1 Test |
|---|---:|---:|---:|---:|
| Linear SVM + TF-IDF | 0.7130 | 0.4810 | 0.6163 | 0.5403 |
| **AraBERT Fine-Tuned** | 0.7667 | 0.5565 | 0.7259 | **0.6300** |
| MARBERTv2 Fine-Tuned | **0.7800** | **0.5855** | 0.6711 | 0.6254 |
| AraBERT Multi-Task (λ = 0.5) | 0.7570 | 0.5401 | **0.7552** | 0.6298 |

### Results Interpretation

- **Best Validation F1:** AraBERT Multi-Task (0.6730).
- **Best Official Test F1:** AraBERT Fine-Tuned (0.6300).
- **Best Official Test Recall:** AraBERT Multi-Task (0.7552).
- **Best Official Test Accuracy and Precision:** MARBERTv2 (0.7800 and 0.5855).

AraBERT Fine-Tuned and AraBERT Multi-Task achieve almost identical official test F1-scores.

Therefore, the experiments do not demonstrate a clear Test F1 advantage for Multi-Task Learning.

No statistical significance test has been conducted to establish whether the small observed difference is meaningful.

---

## 5. Robustness and Generalization

The robustness experiments primarily use the **Linear SVM + TF-IDF baseline**, unless otherwise specified.

### 5.1 Official Unseen Test Evaluation

The trained baseline and fine-tuned Transformer models were evaluated on the official ArSarcasm-v2 test set.

This evaluation measures performance on tweets that were not used during model training or validation-based selection.

### 5.2 Prior Shift Analysis

The proportion of sarcastic tweets was compared across Training, Validation, and Official Test.

| Dataset | Sarcasm Rate |
|---|---:|
| Training | 17.28% |
| Validation | 17.29% |
| Official Test | 27.37% |

The official test set contains approximately **10 percentage points more sarcastic tweets** than the training data.

This difference in class prevalence can influence model evaluation and should be considered when interpreting test performance.

However, class prevalence differences alone do not establish the full cause of any performance change.

### 5.3 Unseen Dialect Evaluation

A Leave-One-Dialect-Out experiment was designed for Egyptian Arabic.

The evaluation procedure is:

1. Exclude Egyptian tweets from the training data.
2. Train a new Linear SVM model using the remaining dialects.
3. Evaluate it on Egyptian tweets from the official test set.
4. Compare its performance with the model trained on all available dialects.

This experiment investigates cross-dialect generalization.

Previous experimental results indicated a decrease in F1 when Egyptian data was excluded from training. The final notebook retains the evaluation code, and the experiment should be rerun to regenerate and verify its exported results.

### 5.4 Shortcut Analysis

The baseline was evaluated on subsets characterized by the presence or absence of URLs and user mentions.

Historical baseline results:

| Feature | Presence | F1 Test |
|---|---|---:|
| URL | With | 0.1026 |
| URL | Without | 0.5496 |
| Mention | With | 0.0000 |
| Mention | Without | 0.5406 |

The URL-containing subset shows substantially lower performance.

The mention-containing subset is very small, so its F1-score should be interpreted cautiously.

These observations identify subgroup performance differences but do not prove that the model causally relies on URLs or mentions.

### 5.5 Error Analysis

False Positives and False Negatives were extracted from the official Linear SVM predictions.

The analysis investigates potential difficulties such as:

- Implicit sarcasm
- Context-dependent irony
- Dialect-specific expressions
- Ambiguous sentiment
- Annotation inconsistencies
- Misleading lexical cues

The notebook includes code to extract 15 False Positives and 15 False Negatives for qualitative inspection.

Manual linguistic annotation of these examples remains part of the final analysis.

---

## 6. Technologies and Tools

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP | NLTK, AraVec |
| Transformers | Hugging Face Transformers, AraBERT, MARBERTv2 |
| Visualization | Matplotlib, Seaborn |
| Experimentation | Google Colab, Kaggle |
| GPU | NVIDIA Tesla T4 |
| Version Control | Git, GitHub |

---

## 7. Project Structure

The repository contains the original experiments and the reorganized final notebook.

```text
Arabic-Sarcasm-Detection/
│
├── README.md
├── compte_rendu.md
│
├── TP2_Arabic_Sarcasm.ipynb
├── TP2_Arabic_Sarcasm_FINAL.ipynb
├── arabic-sarcasm-final-organized.ipynb
│
└── ... additional project files
```

**Main Notebook:** `arabic-sarcasm-final-organized.ipynb`

The organized notebook contains the experimental sections, saved training outputs, model comparisons, and robustness evaluation code.

The original notebooks are preserved for experiment traceability.

Transformer checkpoints are stored separately and are not included in the GitHub repository because of their size.

---

## 8. How to Run

### Option 1 — Explore Existing Results

1. Open `arabic-sarcasm-final-organized.ipynb` on GitHub or in Jupyter.
2. Navigate through Parts A–G.
3. Review the saved training outputs and official evaluation results.
4. Consult the final comparison tables and conclusions.

**No model retraining is required to review the saved results.**

### Option 2 — Reproduce Experiments

1. Open the notebook in Kaggle or Google Colab.
2. Install the required dependencies.
3. Load ArSarcasm-v2.
4. Execute data preprocessing.
5. Run the selected experimental sections.
6. Evaluate models and export results.

A GPU is recommended for Transformer fine-tuning.

Running the entire notebook may retrain large models and require substantial computational resources.

### Checkpoint Management

Best Transformer checkpoints were saved for:

- AraBERT Fine-Tuned
- MARBERTv2 Fine-Tuned
- AraBERT Multi-Task (λ = 0.5)

These checkpoints can be reused for inference and evaluation without repeating training, provided the appropriate model architecture and tokenizer are restored.

---

## 9. Key Findings

**1. Fine-tuned Transformers outperform the classical baseline.**

All three evaluated fine-tuned Transformers achieved higher official Test F1-scores than Linear SVM.

**2. Multi-Task Learning achieves the highest Validation F1.**

AraBERT Multi-Task reached 0.6730 on the latest Kaggle validation experiment.

**3. The best Validation model is not necessarily the best Test model.**

AraBERT Mono-Task achieved a marginally higher Test F1 than Multi-Task.

**4. Multi-Task improves recall among the evaluated models.**

AraBERT Multi-Task obtained the highest observed Test Recall (0.7552), but also produced more False Positives than the other fine-tuned Transformers.

**5. Dataset characteristics matter.**

Differences in sarcasm prevalence, dialect distribution, and linguistic features may affect generalization.

**6. Robustness analysis complements global metrics.**

Subgroup evaluation and qualitative error analysis provide additional information beyond overall Accuracy and F1-score.

---

## 10. Limitations

- Most experiments use a single Train/Validation split.
- Multiple training seeds were not systematically evaluated.
- Not all historical Deep Learning models were evaluated on the official test set.
- Most dialect and shortcut robustness analyses focus on Linear SVM.
- The latest Transformer experiments do not include statistical significance testing.
- Some robustness outputs and qualitative error annotations require final verification.
- The official test set has a different sarcasm prevalence from the training data.

---

## 11. Future Improvements

Potential extensions include:

- Evaluating robustness across several random seeds
- Applying Bootstrap Confidence Intervals and McNemar's Test
- Testing Transformers under Leave-One-Dialect-Out conditions
- Investigating Multi-Task loss weights other than λ = 0.5
- Conducting deeper qualitative error analysis
- Measuring subgroup performance for all fine-tuned Transformers
- Exploring data augmentation for dialectal Arabic

---

## 12. Conclusion

This project presents a comparative experimental study of Arabic sarcasm detection using traditional Machine Learning, Deep Learning, and pretrained Transformer architectures.

The experiments demonstrate the effectiveness of contextualized Arabic language representations for detecting sarcasm in tweets.

**AraBERT Multi-Task achieved the highest Validation F1-score (0.6730), while AraBERT Fine-Tuned achieved the highest observed Official Test F1-score (0.6300).**

The results show that improved validation performance does not necessarily translate into better generalization on unseen data.

Furthermore, the robustness analyses highlight the importance of considering class distribution, dialect variation, linguistic cues, and prediction errors when evaluating NLP systems for Arabic sarcasm detection.

---

## Author

**Oumayma Zerrik**  
Master's Student — Big Data & Internet of Things  
ENSAM Casablanca, Morocco

**Areas of Interest:** Natural Language Processing, Machine Learning, Deep Learning, Transformers, and Artificial Intelligence.

---

## License

This project was developed for academic and educational purposes. The applicable dataset, pretrained model, and third-party software licenses should be respected.
