# Credit Card Fraud Detection — Deployable Project

A complete pipeline: train in Google Colab → deploy a live app on Streamlit
Community Cloud. The app UI is a dark "security command-center" theme
(radar-sweep header, live risk gauge, color-coded verdicts).

## Files

| File | Purpose | Push to GitHub? |
|---|---|---|
| `app.py` | Streamlit app — single & batch fraud predictions, full custom UI | ✅ Yes |
| `requirements.txt` | Python dependencies for the Streamlit app | ✅ Yes |
| `.streamlit/config.toml` | Theme colors (dark navy + cyan accent) | ✅ Yes — keep the `.streamlit` folder name exactly |
| `fraud_detection_model.pkl` | Trained classifier | ✅ Yes |
| `scaler_amount.pkl` | Fitted `RobustScaler` for `Amount` | ✅ Yes |
| `scaler_time.pkl` | Fitted `RobustScaler` for `Time` | ✅ Yes |
| `feature_columns.pkl` | Exact column order the model expects | ✅ Yes |
| `Credit_Card_Fraud_Detection_Colab.ipynb` | Training notebook | Optional — nice to keep in the repo for reference, not required for the app to run |
| `creditcard.csv` / `card-fraud.zip` / `sample_data/` / `.config/` / `.ipynb_checkpoints/` | Colab's own working files | ❌ No — don't push these. Not needed at inference time and the dataset is large |

**From your Colab file browser screenshot**, the only 4 files you need to
download are: `feature_columns.pkl`, `fraud_detection_model.pkl`,
`scaler_amount.pkl`, `scaler_time.pkl` — right-click each → **Download**.
Skip `card-fraud.zip`, `sample_data`, `.config`, `.ipynb_checkpoints`.

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

## Step 2 — Arrange your project folder

Your local folder should look like this before you do anything else:

```
fraud-app/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── fraud_detection_model.pkl
├── scaler_amount.pkl
├── scaler_time.pkl
└── feature_columns.pkl
```

Copy the 4 `.pkl` files you downloaded from Colab into this folder
alongside `app.py`, `requirements.txt`, and the `.streamlit/config.toml`
theme file.

## Step 3 — Test locally (optional but recommended)

```bash
cd fraud-app

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`. You should see the dark
"Fraud Radar" UI with the sidebar nav, threshold slider, and radar-sweep
header.

## Step 4 — Push to GitHub

```bash
git init
git add app.py requirements.txt .streamlit *.pkl
git commit -m "Fraud detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> `.pkl` files are usually small enough for a normal GitHub push (a few
> MB). If `fraud_detection_model.pkl` is unusually large (tens of MB,
> e.g. a big Random Forest), GitHub will warn you — in that case reduce
> `n_estimators`/`max_depth` in the notebook and re-save, or use
> [Git LFS](https://git-lfs.com/).

## Step 5 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repo, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**.
5. Streamlit builds the environment from `requirements.txt` and picks up
   `.streamlit/config.toml` automatically for the theme. You'll get a
   public URL like `https://<your-app>.streamlit.app`.

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
