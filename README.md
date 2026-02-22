# NoShowAI — Clinical Appointment No-Show Prediction

An AI-powered healthcare analytics project that predicts patient appointment no-shows using machine learning and provides risk insights through an interactive dashboard.

## Problem

Missed clinical appointments lead to:

- Resource wastage
- Increased waiting times
- Revenue loss
- Disrupted care continuity

## Solution

This system estimates the probability that an appointment will be a no-show using historical scheduling data, helping clinics identify high-risk appointments early and take preventive actions (reminders, rescheduling, outreach).

## Key Features

- Data preprocessing and feature engineering
- Supervised ML-based no-show prediction
- Risk probability estimation per appointment
- Insights into key contributing factors
- Interactive dashboard layer (planned in `ui/`)

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Streamlit
- joblib

Dependencies are listed in `requirement.txt`.

## Architecture

Data (raw/processed) -> Preprocessing/Feature Engineering -> Model Training/Evaluation -> Saved Model -> Prediction/Insights -> Dashboard (Streamlit)

## Project Structure

```
data/           # datasets (raw/processed)
notebooks/      # EDA, preprocessing, modeling notebooks
src/            # reusable preprocessing/training/inference code
model/          # saved trained model artifacts
ui/             # Streamlit dashboard
docs/           # documentation and architecture notes
requirement.txt # Python dependencies
README.md
```

## Team Roles

- Harsh — Data preprocessing & feature engineering
- Manmath — ML modeling & evaluation
- Ansh — Analysis, visualization & insights
- Yash — UI development & system integration

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirement.txt
```

### 3) Run the dashboard

When the Streamlit app entrypoint is added under `ui/` (commonly `ui/app.py`), run:

```bash
streamlit run ui/app.py
```

## Future Scope

- Automated intervention recommendations (reminders / rescheduling suggestions)
- Explainability improvements (e.g., feature importance + per-patient rationale)
- Deployment for clinic workflow integration

## License

Developed for academic and research purposes.
