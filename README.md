
# Arabic Sarcasm Detection — NLP & Deep Learning

## Project Overview

This project explores Arabic sarcasm detection using traditional Machine Learning, Deep Learning, and Pre-trained Language Models (PLMs).

The objective is to compare different NLP approaches, evaluate their effectiveness on Arabic tweets, and analyze model robustness across dialects and linguistic characteristics.

**Dataset:** ArSarcasm-v2  
**Task:** Binary Classification — Sarcastic vs. Non-Sarcastic Tweets  
**Environment:** Python, Google Colab, PyTorch, Scikit-learn, Hugging Face Transformers

---

## 1. Dataset

The project uses **ArSarcasm-v2**, an Arabic Twitter dataset containing sarcasm, sentiment, and dialect annotations.

| Dataset | Number of Tweets |
|---|---:|
| Official Training Set | 12,548 |
| Official Test Set | 3,000 |

The official training set is divided into:

- 90% Training
- 10% Validation
- Stratified split with `random_state=42`

The official test set is reserved for final evaluation.

---

## 2. Text Preprocessing

Arabic text preprocessing includes:

- URL and user mention normalization
- Arabic character normalization
- Diacritics and Tatweel removal
- Whitespace cleaning
- Stop-word removal while preserving negation terms
- Tokenization and sequence preparation

Preprocessing varies depending on model requirements.

---

## 3. Models Implemented

### Traditional Machine Learning

- Linear SVM with Word-Level TF-IDF
- Character-Level TF-IDF
- Combined Word and Character N-grams

### Zero-Shot and Sentiment-Based Methods

- Multilingual NLI Zero-Shot Classification
- CAMeLBERT Sentiment-Based Heuristic

### Deep Learning

- BiLSTM with Random Embeddings
- BiLSTM with AraVec Frozen Embeddings
- BiLSTM with AraVec Trainable Embeddings
- BiLSTM with Attention
- Mini-Transformer with Positional Encoding
- Mini-Transformer without Positional Encoding

### Pre-trained Language Models

- AraBERT Frozen + CLS
- AraBERT Frozen + Mean Pooling
- AraBERT Frozen + BiLSTM + Attention
- AraBERT Fine-Tuning
- MARBERTv2 Fine-Tuning
- AraBERT Multi-Task Learning

---

## 4. Experimental Results

### Validation Performance

The main evaluation metric is the F1-score for the sarcastic class.

| Model | F1 Validation |
|---|---:|
| **AraBERT Multi-Task (λ = 0.5)** | **0.6788** |
| AraBERT Fine-Tuned | 0.6601 |
| MARBERTv2 Fine-Tuned | 0.6578 |
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

**Best validation model:** AraBERT Multi-Task with λ = 0.5.

### Official Test Performance

| Model | Accuracy | Precision | Recall | F1 Test |
|---|---:|---:|---:|---:|
| Linear SVM + TF-IDF | 0.7130 | 0.4810 | 0.6163 | **0.5403** |

The other models were not evaluated on the official test set because their trained checkpoints were unavailable following a Google Colab runtime reset.

Validation results should not be interpreted as official test performance.

---

## 5. Robustness and Generalization

### Prior Shift

The sarcasm class distribution was compared between training, validation, and official test datasets.

This analysis investigates whether changes in class prevalence affect model performance.

### Unseen Dialect Evaluation

A Leave-One-Dialect-Out experiment was conducted using the Egyptian dialect.

The model was trained without Egyptian tweets and evaluated on Egyptian tweets from the official test set.

This experiment examines cross-dialect generalization.

### Shortcut Analysis

Model performance was evaluated according to the presence of URLs and user mentions.

| Feature | Presence | F1 Test |
|---|---|---:|
| URL | With | 0.1026 |
| URL | Without | 0.5496 |
| Mention | With | 0.0000 |
| Mention | Without | 0.5406 |

The results indicate substantial performance differences between URL-containing and URL-free tweets.

The mention-containing subset is very small, so its results should be interpreted cautiously.

### Error Analysis

False Positives and False Negatives were extracted from the official test predictions.

The analysis focuses on:

- Implicit sarcasm
- Context-dependent irony
- Dialect-specific expressions
- Ambiguous sentiment
- Annotation noise
- Misleading lexical cues

---

## 6. Technologies

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP | NLTK, AraVec |
| Transformers | Hugging Face, AraBERT, MARBERTv2 |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab |

---

## 7. Project Structure

```text
Arabic-Sarcasm-Detection/
|
|-- README.md
|-- compte_rendu.md
|-- TP2_Arabic_Sarcasm_FINAL.ipynb
```

---

## 8. How to Run

1. Open `TP2_Arabic_Sarcasm_FINAL.ipynb` in Google Colab.
2. Install the required Python dependencies.
3. Load the ArSarcasm-v2 dataset.
4. Execute the preprocessing and experimental sections.
5. Review the evaluation metrics and visualizations.

**Note:** Reproducing all Deep Learning and Transformer experiments requires substantial computational resources. A GPU is recommended.

---

## 9. Key Findings

- Pre-trained Arabic language models achieved higher validation F1-scores than the implemented classical baselines.
- AraBERT Multi-Task achieved the highest validation F1-score: **0.6788**.
- Linear SVM achieved **0.5403 F1** on the official test set.
- The experiments revealed performance differences related to URL presence.
- Dialect variation remains an important challenge for Arabic sarcasm detection.

---

## 10. Limitations

- Transformer checkpoints were lost following a Google Colab runtime reset.
- Official test scores are unavailable for most Deep Learning models.
- Some robustness analyses were performed only on the Linear SVM baseline.
- Validation performance alone does not establish superiority on unseen test data.

---

## 11. Conclusion

This project demonstrates the potential of pre-trained Arabic language models for sarcasm detection.

AraBERT Multi-Task delivered the strongest validation results, while the Linear SVM baseline provided a reproducible official test evaluation.

The experiments highlight the importance of dialect-aware evaluation, robustness analysis, and careful separation between validation and test performance.

---

## Author

**Oumayma Zerrik**  
Master's Student in Big Data & IoT  
ENSAM Mohammedia

**Fields:** Natural Language Processing, Machine Learning, Deep Learning, and Artificial Intelligence.

---

## License

This project was developed for academic and educational purposes.
