# PhishGuard-XAI
Explainable, drift-aware phishing URL detection (research project code).

## Pipeline
URL -> lexical feature extraction -> drift monitor (PSI + KS) -> XGBoost classifier -> SHAP explanation -> drift-triggered weighted retraining.

## Setup
```bash
git clone https://github.com/<your-username>/PhishGuard-XAI.git
cd PhishGuard-XAI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
# 1. Smoke test on SYNTHETIC data (figures are watermarked; never report these numbers)
python run_experiments.py --demo

# 2. REAL experiments (needed for the paper)
#    CSVs need columns: url,label   (label: 1 = phishing, 0 = legitimate)
python run_experiments.py --train data/train.csv --drift data/later_period.csv

# 3. Explain one URL
python -m src.app "http://example-login.xyz/verify"
```
Outputs (tables .csv + figures .png) are written to `outputs/`.

## Using real data
Use a public, citable phishing-URL dataset and split it **by time** (older = train, newer = drift set) so that the drift experiment is genuine. Record dataset name, version, licence and download date in the paper.

## Layout
`src/features.py` feature extractor · `src/drift.py` PSI/KS monitor · `src/models.py` baselines · `src/explain.py` SHAP · `run_experiments.py` full pipeline · `outputs/` generated results
