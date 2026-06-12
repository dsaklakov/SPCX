# SPCX Live Sale Model

Public Streamlit dashboard for SPCX live price, sale ladder, risk model, float math, turnover, implied market cap, and Excel export.

## Public deployment target

This repo is designed for Streamlit Community Cloud. After deployment, the app can be shared as a public link and opened by anyone.

## What the app does

- Pulls quote data from Polygon when `POLYGON_API_KEY` is configured.
- Falls back to Yahoo/yfinance if no Polygon key is configured.
- Supports manual override if quote APIs fail.
- Recalculates the sale ladder for 160, 170, 190, 210, 250, 350, 450, 500.
- Calculates current price, intraday high, intraday low, open, volume, dollar volume, equilibrium, sale point, stop / invalidation, risk to stop, reward/risk, IPO float, shares still held, turnover, and implied market cap.
- Shows a scenario chart.
- Exports the current model to Excel.

## Streamlit Cloud deployment

1. Go to Streamlit Community Cloud.
2. Create a new app from this GitHub repository.
3. Repository: `dsaklakov/SPCX`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Set visibility to public.
7. Deploy.

Optional but recommended: add `POLYGON_API_KEY` in Streamlit secrets for better live/near-live quotes.

Without Polygon, the app still works through Yahoo/yfinance fallback.

## Streamlit secrets

In Streamlit Cloud, open app settings and add:

```toml
POLYGON_API_KEY = "your_polygon_key_here"
```

Do not commit a real API key to GitHub.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
