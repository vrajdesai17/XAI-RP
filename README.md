# XAI for Detection of Fake News & Hate Speech

A peer-reviewed research project applying **Explainable AI (XAI)** techniques to a transformer-based classifier for joint detection of fake news and hate speech. Fine-tunes **DistilBERT** on the **HateXplain** dataset and layers **SHAP**, **LIME**, and **ELI5** to surface token-level attribution maps that explain *why* the model makes each prediction.

> 📄 **Published Paper:** [https://www.taylorfrancis.com/chapters/edit/10.1201/9781003409519-6/explainable-models-detection-incidents-fake-news-hate-speech-vraj-desai-ashray-gattani-harshal-dalvi?context=ubx&refId=1a813f28-767d-4700-886f-9b56f5a6779c]

---

## Why this project

Black-box transformer classifiers achieve strong accuracy on toxicity detection but offer no insight into their reasoning. For content moderation — where false positives hurt free expression and false negatives leave users exposed to harm — explainability is essential.

This project builds an end-to-end pipeline that doesn't just classify hate speech and fake news — it explains every decision through three complementary attribution methods, enabling moderators and researchers to audit the model's behavior.

---

## Results

- **91% F1 Score** on the HateXplain test set (binary classification: toxic vs. non-toxic)
- **20K+ annotated samples** trained with class-weighted loss to handle label imbalance across 3 categories (hateful, offensive, normal)
- **Token-level explanations** generated for every prediction via SHAP, LIME, and ELI5

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | DistilBERT (via Hugging Face Transformers) |
| Framework | TensorFlow / PyTorch |
| Dataset | HateXplain (~20K annotated posts) |
| Explainability | SHAP, LIME, ELI5 |
| Language | Python 3.9+ |
| Environment | Jupyter Notebook |

---

## Methodology

### 1. Data Preparation
- HateXplain dataset (~20K posts labeled across 3 classes: hateful, offensive, normal)
- Class-weighted loss to handle imbalanced label distribution
- 80/10/10 train/val/test split (stratified)

### 2. Model Fine-Tuning
- Base model: `distilbert-base-uncased`
- Sequence length: 128 tokens
- Fine-tuned with cross-entropy loss + class weights
- Optimizer: AdamW, learning rate 2e-5
- Evaluation: precision, recall, F1 (weighted)

### 3. Explainability Layer
Three complementary attribution methods applied to model predictions:
- **SHAP** — game-theoretic Shapley value attribution for global + local importance
- **LIME** — local surrogate models for per-sample explanations
- **ELI5** — token-weight visualization for prediction transparency

Each method generates a heatmap highlighting which tokens drove the classification, displayed in the dashboard for human auditing.

---

## Repository Structure
XAI-RP/
├── XAI models.ipynb     # Main notebook: fine-tuning + explainability
├── test.csv             # Held-out test split
└── README.md

---

## Getting Started

### Prerequisites
- Python 3.9+
- 4GB+ GPU recommended for fine-tuning (CPU works but slow)

### Install dependencies
```bash
pip install transformers tensorflow shap lime eli5 scikit-learn pandas numpy matplotlib
```

### Run the notebook
```bash
jupyter notebook "XAI models.ipynb"
```

The notebook walks through:
1. Loading and preprocessing HateXplain
2. Fine-tuning DistilBERT
3. Evaluating on the test set
4. Generating SHAP / LIME / ELI5 explanations

---

## Sample Output

Given the input `"go back to your country"`, the model classifies it as **hateful** with high confidence. The explainability layer surfaces:
- **SHAP** weights the phrase `"back to your country"` most heavily
- **LIME** highlights `"go back"` as the second strongest signal
- **ELI5** shows token-by-token contribution scores in a color-graded view

All three converge on the same underlying signal — building trust in the model's reasoning rather than treating it as a black box.

---

## Citation

If you use this work, please cite:

```bibtex
@article{desai2023xai,
  title   = {XAI for Detection of Incidents of Fake News and Hate Speech},
  author  = {Desai, Vraj and [co-authors]},
  journal = {[Journal name]},
  year    = {2023}
}
```

---

## License

MIT

---

## Author

**Vraj Desai** — [GitHub](https://github.com/vrajdesai17) · [LinkedIn](https://linkedin.com/in/vrajdesai17)
