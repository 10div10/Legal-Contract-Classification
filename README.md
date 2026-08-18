# Legal Contract Type Classification — Classical ML Baselines

A reproducible legal-NLP baseline that classifies commercial contract text using TF-IDF and classical machine-learning models.

The project compares six standard ML classifiers under the same stratified 5-fold cross-validation setup, then tests an enhanced word + character TF-IDF Linear SVM baseline.

## Contract classes

The current benchmark uses seven agreement types that are sufficiently represented in CUAD:

- Collaboration Agreement
- Development Agreement
- Distributor Agreement
- License Agreement
- Maintenance Agreement
- Service Agreement
- Sponsorship Agreement

## Why these seven instead of NDA / Employment / Lease / Privacy Policy / Purchase / License / Service?

The original project idea used those seven labels, but CUAD does not contain enough clean examples of several of them for an honest 7-class benchmark. Rather than generate synthetic legal documents or report misleading metrics, this repository uses seven well-supported real CUAD contract types.

The pipeline is label-agnostic: replacing the source dataset and `CLASS_PATTERNS` allows the original target taxonomy to be used later.

## Dataset

Source: **Contract Understanding Atticus Dataset (CUAD)** by The Atticus Project.

- 510 commercial contracts
- Expert annotations for 41 contract-review clause categories
- CC BY 4.0
- Dataset: https://huggingface.co/datasets/theatticusproject/cuad
- Project: https://www.atticusprojectai.org/datasets/
- Paper: https://arxiv.org/abs/2103.06268

For this baseline, the agreement label is derived from the source filename. The model input **does not use the filename, document-name field, or answer fields**. Instead, the available annotated clause-context text is concatenated into one sample per contract. This reduces direct label leakage, but it is still a proxy for full-document classification and should be treated as a baseline experiment rather than a production benchmark.

After filtering very short samples, the benchmark contains **230 contracts**.

## Models compared

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM
4. SGD Classifier
5. Random Forest
6. XGBoost
7. Linear SVM with combined word + character TF-IDF

## Current results

Stratified 5-fold cross-validation, random state 42.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM + word/char TF-IDF | **0.7000** | **0.7253** | **0.7020** | **0.6984** | **0.6973** |
| Logistic Regression | 0.6609 | 0.6939 | 0.6590 | 0.6596 | 0.6614 |
| Linear SVM | 0.6609 | 0.6862 | 0.6576 | 0.6568 | 0.6594 |
| XGBoost | 0.5783 | 0.5889 | 0.5791 | 0.5713 | 0.5684 |
| SGD Classifier | 0.5870 | 0.6560 | 0.5887 | 0.5708 | 0.5711 |
| Random Forest | 0.5957 | 0.5924 | 0.5825 | 0.5666 | 0.5748 |
| Multinomial NB | 0.4696 | 0.5801 | 0.4309 | 0.4219 | 0.4377 |

> These are baseline results on a small CUAD-derived proxy dataset. They are not claims of production legal-document accuracy.

## Best model class-level F1

Out-of-fold predictions from the enhanced Linear SVM:

| Class | F1 |
|---|---:|
| Sponsorship Agreement | 0.8333 |
| Distributor Agreement | 0.8286 |
| License Agreement | 0.7647 |
| Service Agreement | 0.6842 |
| Maintenance Agreement | 0.6429 |
| Collaboration Agreement | 0.5823 |
| Development Agreement | 0.5490 |

## Project structure

```text
legal-contract-classifier/
├── src/
│   ├── prepare_data.py
│   ├── train_baselines.py
│   └── predict.py
├── data/
├── results/
├── models/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py
python src/train_baselines.py
```

The training script creates:

- `results/model_comparison.csv`
- `results/best_model_classification_report.csv`
- `results/confusion_matrix.png`
- `results/run_metadata.json`
- `models/legal_contract_classifier.joblib`

## Predict a new contract

```bash
python src/predict.py path/to/contract.pdf
```

or

```bash
python src/predict.py path/to/contract.txt
```

The SVM outputs decision scores, not calibrated probabilities.

## Methodology

```text
CUAD master clauses
        ↓
Agreement-type label derived from filename
        ↓
Remove filename/title/answer fields from model input
        ↓
Concatenate legal clause-context text
        ↓
TF-IDF features
        ↓
Stratified 5-fold CV
        ↓
Compare classical ML models
        ↓
Enhanced word + character TF-IDF Linear SVM
        ↓
Fit final baseline model
```

## Next improvements

- Train on complete contract text rather than annotated clause-context proxies.
- Build the original target taxonomy: NDA, Employment, Lease, Service, Privacy Policy, Purchase, and License.
- Increase samples per class and use group-aware splitting when related contracts/amendments are present.
- Add Legal-BERT / ModernBERT / transformer baselines.
- Add probability calibration and confidence-based human review.
- Add model-error analysis and explainability using top weighted n-grams.
- Deploy behind FastAPI or Streamlit.

## Important note

This repository is an ML/NLP research project. It does not provide legal advice and should not be used to make legal decisions without qualified human review.

## Licensing

Repository code is MIT licensed. CUAD data is separately licensed under CC BY 4.0 and must retain its required attribution.
