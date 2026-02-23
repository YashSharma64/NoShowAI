# NoShowAI — Clinical Appointment No-Show Prediction

An AI-powered healthcare analytics system that predicts patient appointment no-shows using machine learning and surfaces risk insights through an interactive Streamlit dashboard.

---

## Problem

Missed clinical appointments lead to:

- Resource wastage and idle clinic capacity
- Increased patient waiting times
- Revenue loss for healthcare providers
- Disrupted continuity of patient care

## Solution

NoShowAI estimates the probability that an appointment will result in a no-show using historical scheduling data. Clinics can identify high-risk appointments early and take preventive actions — sending reminders, rescheduling, or initiating social-work outreach.

---

## Key Features

- **Data Preprocessing & Feature Engineering** — cleaned appointment records with lead-time, demographic encoding, and reminder flags
- **Multi-Model ML Training** — Logistic Regression, Decision Tree, and Random Forest trained with class balancing (SMOTE or `class_weight='balanced'`)
- **Best-Model Auto-Selection** — model with highest F1 (No-Show) score is saved as `model/best_model.pkl`
- **Risk Probability Estimation** — per-appointment probability scores with configurable high-risk threshold
- **Interactive Streamlit Dashboard** — 3-page workflow: Upload → Risk Dashboard → Analytics
- **Model Performance Comparison** — Accuracy, F1, and R² metrics surfaced per model in the UI

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| ML | `scikit-learn`, `imbalanced-learn` (optional, for SMOTE) |
| Serialisation | `joblib` |
| Visualisation | `matplotlib`, `seaborn`, Streamlit native charts |
| Dashboard | `streamlit` |

Dependencies: `requirement.txt`

---

## Architecture

```
Raw Data (CSV)
    │
    ▼
Preprocessing / Feature Engineering  ←── notebooks/
    │
    ▼
Model Training  (src/train.py)
    ├── Logistic Regression   →  model/logistic_regression.pkl
    ├── Decision Tree         →  model/decision_tree.pkl
    ├── Random Forest         →  model/random_forest.pkl
    └── Best Model (by F1)    →  model/best_model.pkl
    │
    ▼
Streamlit Dashboard  (ui/app.py)
    ├── Page 1: Upload Dataset
    ├── Page 2: Risk Dashboard  (predictions + colour-coded risk table)
    └── Page 3: Analytics       (distributions, model metrics, feature importance)
```

---

## Project Structure

```
NoShowAI/
├── data/                   # Raw and processed datasets
├── notebooks/              # EDA, preprocessing, and modelling notebooks
├── src/
│   └── train.py            # Full training pipeline (3 models + best-model export)
├── model/                  # Saved model artefacts (.pkl files)
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── ui/
│   └── app.py              # Streamlit dashboard (Upload → Risk → Analytics)
├── docs/                   # Architecture notes and documentation
├── requirement.txt         # Python dependencies
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

> Optionally install `imbalanced-learn` for SMOTE oversampling during training:
> ```bash
> pip install imbalanced-learn
> ```

### 3. Train the models

Place your cleaned dataset at `data/noshow_cleaned (1).csv`, then run:

```bash
python src/train.py
```

This will train all three models, print a comparison table, and save the best-performing model as `model/best_model.pkl`.

### 4. Launch the dashboard

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser. Follow the 3-step in-app workflow:

1. **Upload Dataset** — upload the preprocessed CSV
2. **Risk Dashboard** — click *Predict No-Show Risk* to score all appointments; high-risk rows are highlighted in red
3. **Analytics** — view model performance (Accuracy / F1 / R²), no-show distribution, probability histogram, and feature importance

---

## Dashboard Workflow

| Step | Page | Action |
|---|---|---|
| 1 | Upload Dataset | Upload a preprocessed appointments CSV |
| 2 | Risk Dashboard | Run predictions; adjust high-risk threshold (default 0.70) |
| 3 | Analytics | Review per-model metrics, charts, and feature importances |

---

## Team Roles

| Member | Responsibility |
|---|---|
| Harsh | Data preprocessing & feature engineering |
| Manmath | ML modelling & evaluation |
| Ansh | Analysis, visualisation & insights |
| Yash | UI development & system integration |

---

## Future Scope

- Automated intervention recommendations (reminders / rescheduling suggestions)
- Explainability layer (SHAP values for per-patient rationale)
- Deployment for clinic workflow integration (Docker / cloud)
- Real-time appointment ingestion via API

---

## License

Developed for academic and research purposes.
