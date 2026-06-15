import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sell Decision Engine", layout="wide")

st.markdown("""
<style>
.stApp { background: #050505; }
[data-testid="stSidebar"] { background: #0B0B0B; }
[data-testid="stMetric"] {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 12px;
    padding: 10px;
}
h1,h2,h3 { color: #E5E5E5 !important; }
p,span,label { color: #CFCFCF !important; }
section[data-testid="stSidebar"] * { color: #D9D9D9 !important; }
</style>
""", unsafe_allow_html=True)

WEIGHTS = {
    "Valuation": 14.0,
    "Expected Upside": 9.0,
    "Momentum": 8.0,
    "Volatility": 5.0,
    "Liquidity": 5.0,
    "Position Size": 9.0,
    "Time Horizon": 10.0,
    "Event Risk": 9.0,
    "Narrative Risk": 8.0,
    "Portfolio Concentration": 13.0,
    "Business Dependency Risk": 10.0,
}


def score_from_thresholds(value: float, thresholds, reverse: bool = False) -> float:
    if value is None or pd.isna(value):
        return 0.0
    scores = [20, 40, 60, 80, 100]
    if reverse:
        for threshold, score in zip(thresholds, scores):
            if value <= threshold:
                return float(score)
        return 0.0
    result = 0.0
    for threshold, score in zip(thresholds, scores):
        if value >= threshold:
            result = float(score)
    return result


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def action_from_score(score: float) -> str:
    if score < 13:
        return "HOLD"
    if score < 21:
        return "MONITOR / PREPARE LIMIT ORDERS"
    if score < 34:
        return "SELL 13% OF POSITION"
    if score < 55:
        return "SELL 21% OF POSITION"
    if score < 89:
        return "SELL 34% OF POSITION"
    return "SELL 55% NOW; PUT REMAINDER UNDER -8% TRAILING STOP"


def recommended_sale_pct(score: float) -> float:
    if score < 21:
        return 0.0
    if score < 34:
        return 13.0
    if score < 55:
        return 21.0
    if score < 89:
        return 34.0
    return 55.0


def weighted_contribution(score: float, weight: float) -> float:
    return score * weight / 100.0


st.title("Sell Decision Engine")
st.caption("Multi-factor exit framework for SPCX or any other volatile asset. This is a decision-support model, not a prediction engine.")

with st.sidebar:
    st.header("Core Inputs")
    ticker = st.text_input("Ticker / Asset", value="SPCX").upper().strip()
    current_price = st.number_input("Current / Close Price", min_value=0.0, value=160.95, step=0.01)
    fair_value = st.number_input("Fair Value", min_value=0.01, value=63.00, step=0.01)

    st.header("Targets")
    targets_text = st.text_input("Sell targets", value="170, 190, 210, 250, 350, 450, 500")
    probabilities_text = st.text_input("Probabilities for targets, %", value="30, 10, 5, 2, 1, 0.5, 0.2")

    st.header("Market Inputs")
    price_5d_ago = st.number_input("Price 5 trading days ago", min_value=0.0, value=135.00, step=0.01)
    vol_10d = st.number_input("10d volatility", min_value=0.0, value=0.40, step=0.01)
    vol_90d = st.number_input("90d volatility", min_value=0.0, value=0.08, step=0.01)
    current_volume = st.number_input("Current daily volume", min_value=0.0, value=315_000_000.0, step=1000.0)
    avg_volume_20d = st.number_input("20d average volume", min_value=0.0, value=31_500_000.0, step=1000.0)
    day_low = st.number_input("Day low", min_value=0.0, value=149.34, step=0.01)
    day_high = st.number_input("Day high", min_value=0.0, value=176.52, step=0.01)

    st.header("Portfolio Inputs")
    position_value = st.number_input("Position value", min_value=0.0, value=200_000.0, step=1000.0)
    total_portfolio_value = st.number_input("Total portfolio value", min_value=0.01, value=1_000_000.0, step=1000.0)
    second_largest_position_value = st.number_input("Second-largest position value", min_value=0.01, value=40_000.0, step=1000.0)
    days_until_cash_needed = st.number_input("Days until cash needed", min_value=0, value=365, step=1)

    st.header("Judgment Inputs")
    event_severity = st.slider("Event severity, 1-10", 1.0, 10.0, 1.0, 0.5)
    event_probability = st.slider("Negative event probability", 0.0, 1.0, 0.0, 0.05)
    days_to_event = st.number_input("Days to event", min_value=0, value=180, step=1)
    media_index_48h = st.number_input("48h media / narrative index", min_value=0.0, value=10.0, step=0.1)
    media_index_90d_avg = st.number_input("90d average media index", min_value=0.01, value=1.0, step=0.1)

    st.header("Business Dependency Risk")
    customer_concentration = st.slider("Customer concentration risk", 0.0, 100.0, 20.0, 1.0)
    product_concentration = st.slider("Product concentration risk", 0.0, 100.0, 60.0, 1.0)
    platform_dependence = st.slider("Platform / supplier dependence risk", 0.0, 100.0, 20.0, 1.0)
    execution_risk = st.slider("Execution risk", 0.0, 100.0, 30.0, 1.0)

try:
    targets = [float(x.strip()) for x in targets_text.split(",") if x.strip()]
    probs = [float(x.strip()) / 100.0 for x in probabilities_text.split(",") if x.strip()]
except Exception:
    st.error("Targets and probabilities must be comma-separated numbers.")
    st.stop()

if len(targets) != len(probs):
    st.error("The number of targets must match the number of probabilities.")
    st.stop()

# Factor 1: Valuation
valuation_pct = ((current_price - fair_value) / fair_value) * 100.0
valuation_score = score_from_thresholds(valuation_pct, [34, 55, 89, 144, 233])

# Factor 2: Expected Upside
expected_upside_dollars = sum(p * max(t - current_price, 0.0) for t, p in zip(targets, probs))
expected_upside_pct = (expected_upside_dollars / current_price) * 100.0 if current_price > 0 else 0.0
expected_upside_score = score_from_thresholds(expected_upside_pct, [-10, 0, 5, 10, 20], reverse=True)

# Factor 3: Momentum
momentum_pct = ((current_price - price_5d_ago) / price_5d_ago) * 100.0 if price_5d_ago > 0 else 0.0
momentum_score = score_from_thresholds(momentum_pct, [21, 34, 55, 89, 144])

# Factor 4: Volatility
vol_ratio = vol_10d / vol_90d if vol_90d > 0 else 0.0
volatility_score = score_from_thresholds(vol_ratio, [1, 1.5, 2, 3, 5])

# Factor 5: Liquidity only counts near the top of the day range
if day_high > day_low:
    price_position = (current_price - day_low) / (day_high - day_low)
else:
    price_position = 0.0
volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0.0
liquidity_score = score_from_thresholds(volume_ratio, [2, 3, 5, 8, 13]) if price_position >= 0.80 else 0.0

# Factor 6: Position Size
position_pct = (position_value / total_portfolio_value) * 100.0 if total_portfolio_value > 0 else 0.0
position_size_score = score_from_thresholds(position_pct, [5, 10, 15, 25, 40])

# Factor 7: Time Horizon
# Continuous score. More urgent cash need means higher sell score.
time_horizon_score = clamp(100.0 * math.exp(-float(days_until_cash_needed) / 90.0))

# Factor 8: Event Risk
# Max raw = 100 when severity=10, probability=1, days=0.
event_raw = 10.0 * event_severity * event_probability * math.exp(-float(days_to_event) / 30.0)
event_risk_score = score_from_thresholds(event_raw, [1, 3, 5, 8, 13])

# Factor 9: Narrative Risk
narrative_ratio = media_index_48h / media_index_90d_avg if media_index_90d_avg > 0 else 0.0
narrative_score = score_from_thresholds(narrative_ratio, [2, 3, 5, 8, 13])

# Factor 10: Portfolio Concentration
portfolio_share_score = score_from_thresholds(position_pct, [10, 20, 30, 50, 80])
relative_concentration = position_value / second_largest_position_value if second_largest_position_value > 0 else 0.0
relative_concentration_score = score_from_thresholds(relative_concentration, [1.5, 2, 3, 5, 8])
portfolio_concentration_score = 0.60 * portfolio_share_score + 0.40 * relative_concentration_score

# Factor 11: Business Dependency Risk
business_dependency_raw = (
    0.30 * customer_concentration
    + 0.30 * product_concentration
    + 0.20 * platform_dependence
    + 0.20 * execution_risk
)
business_dependency_score = score_from_thresholds(business_dependency_raw, [20, 40, 60, 80, 100])

factor_rows = [
    ["Valuation", WEIGHTS["Valuation"], valuation_score, weighted_contribution(valuation_score, WEIGHTS["Valuation"]), f"Overvaluation: {valuation_pct:.1f}%"],
    ["Expected Upside", WEIGHTS["Expected Upside"], expected_upside_score, weighted_contribution(expected_upside_score, WEIGHTS["Expected Upside"]), f"Expected upside: {expected_upside_pct:.1f}%"],
    ["Momentum", WEIGHTS["Momentum"], momentum_score, weighted_contribution(momentum_score, WEIGHTS["Momentum"]), f"5d momentum: {momentum_pct:.1f}%"],
    ["Volatility", WEIGHTS["Volatility"], volatility_score, weighted_contribution(volatility_score, WEIGHTS["Volatility"]), f"Vol ratio 10d/90d: {vol_ratio:.2f}"],
    ["Liquidity", WEIGHTS["Liquidity"], liquidity_score, weighted_contribution(liquidity_score, WEIGHTS["Liquidity"]), f"Volume ratio: {volume_ratio:.2f}; day-range position: {price_position:.1%}"],
    ["Position Size", WEIGHTS["Position Size"], position_size_score, weighted_contribution(position_size_score, WEIGHTS["Position Size"]), f"Position / portfolio: {position_pct:.1f}%"],
    ["Time Horizon", WEIGHTS["Time Horizon"], time_horizon_score, weighted_contribution(time_horizon_score, WEIGHTS["Time Horizon"]), f"Days until cash needed: {days_until_cash_needed}"],
    ["Event Risk", WEIGHTS["Event Risk"], event_risk_score, weighted_contribution(event_risk_score, WEIGHTS["Event Risk"]), f"Event raw: {event_raw:.2f}"],
    ["Narrative Risk", WEIGHTS["Narrative Risk"], narrative_score, weighted_contribution(narrative_score, WEIGHTS["Narrative Risk"]), f"Media ratio: {narrative_ratio:.2f}"],
    ["Portfolio Concentration", WEIGHTS["Portfolio Concentration"], portfolio_concentration_score, weighted_contribution(portfolio_concentration_score, WEIGHTS["Portfolio Concentration"]), f"Portfolio share: {position_pct:.1f}%; vs #2: {relative_concentration:.2f}x"],
    ["Business Dependency Risk", WEIGHTS["Business Dependency Risk"], business_dependency_score, weighted_contribution(business_dependency_score, WEIGHTS["Business Dependency Risk"]), f"Raw dependency score: {business_dependency_raw:.1f}"],
]

factor_df = pd.DataFrame(factor_rows, columns=["Factor", "Weight %", "Score", "Contribution", "Diagnostic"])
sell_score = float(factor_df["Contribution"].sum())
action = action_from_score(sell_score)
sale_pct = recommended_sale_pct(sell_score)
trailing_stop = current_price * 0.92

k1, k2, k3, k4 = st.columns(4)
k1.metric("SELL_SCORE", f"{sell_score:.2f}/100")
k2.metric("Action", action)
k3.metric("Sale size", f"{sale_pct:.0f}%")
k4.metric("-8% trailing stop", f"${trailing_stop:,.2f}")

st.subheader("Factor Contributions")
st.dataframe(
    factor_df.style.format({
        "Weight %": "{:.1f}",
        "Score": "{:.1f}",
        "Contribution": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Contribution Chart")
st.bar_chart(factor_df.set_index("Factor")["Contribution"])

st.subheader("Target Distribution Used for Expected Upside")
target_df = pd.DataFrame({
    "Target": targets,
    "Probability": probs,
    "Upside $/share": [max(t - current_price, 0.0) for t in targets],
    "Probability-weighted upside $/share": [p * max(t - current_price, 0.0) for t, p in zip(targets, probs)],
})
st.dataframe(
    target_df.style.format({
        "Target": "${:,.2f}",
        "Probability": "{:.2%}",
        "Upside $/share": "${:,.2f}",
        "Probability-weighted upside $/share": "${:,.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption("Business Dependency Risk is generic: customer concentration, product concentration, platform/supplier dependence and execution risk. It replaces single-company fields such as NASA dependence so the engine can later support other assets.")
