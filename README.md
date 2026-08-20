# Credit Card Fraud Detection Model
A complete pipeline: train in Google Colab → deploy a live app on Streamlit
Community Cloud.

## Files

| File | Purpose |
|---|---|
| `Credit_Card_Fraud_Detection_Colab.ipynb` | Full training notebook (EDA, preprocessing, SMOTE, model training/comparison, saves artifacts) |
| `app.py` | Streamlit app — single & batch fraud predictions |
| `requirements.txt` | Python dependencies for the Streamlit app |
| `fraud_detection_model.pkl`, `scaler_amount.pkl`, `scaler_time.pkl`, `feature_columns.pkl` | Model artifacts — **produced by running the notebook**, not included here |

## Step 1 — Train the model in Colab

1. Open `Credit_Card_Fraud_Detection_Colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells top to bottom.
3. When prompted, upload `creditcard.csv` (the Kaggle Credit Card Fraud
   Detection dataset — search "Credit Card Fraud Detection" on
   [kaggle.com/datasets](https://www.kaggle.com/datasets) if you don't
   already have it).
4. At the end, the notebook automatically downloads 4 files to your
   computer:
   - `fraud_detection_model.pkl`
   - `scaler_amount.pkl`
   - `scaler_time.pkl`
   - `feature_columns.pkl`

## Step 2 — Test locally (optional but recommended)

```bash
mkdir fraud-app && cd fraud-app
# copy in: app.py, requirements.txt, and the 4 .pkl files from Step 1

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Step 3 — Push to GitHub

```bash
git init
git add app.py requirements.txt *.pkl
git commit -m "Fraud detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> Note: `creditcard.csv` itself does **not** need to go in the repo —
> only `app.py`, `requirements.txt`, and the 4 `.pkl` files. Keep the raw
> dataset out of GitHub (it's large and not needed at inference time).

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repo, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**.
5. Streamlit builds the environment from `requirements.txt` and gives you
   a public URL like `https://<your-app>.streamlit.app`.

## Using the app

- **Single Transaction** — manually enter `Time`, `Amount`, and (optionally)
  the `V1`–`V28` PCA features to check one transaction.
- **Batch Prediction (CSV)** — upload a CSV with columns
  `Time, V1, ..., V28, Amount` (and optionally `Class` for ground-truth
  evaluation) to score many transactions at once and download the results.
- Adjust the **fraud decision threshold** in the sidebar to trade off
  catching more fraud vs. fewer false alarms.

## Notes on model choice

The notebook trains and compares **Logistic Regression** and
**Random Forest**, and saves whichever you set as `final_model` (Random
Forest by default). Class imbalance is handled with **SMOTE** applied only
to the training set — the test set is left untouched so evaluation
reflects real-world performance, not an artificially balanced sample.
