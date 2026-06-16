import math
import os
from datetime import datetime, time, timezone
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from sell_engine import (
        score_valuation,
        score_expected_upside,
        score_momentum,
        score_volatility_ratio,
        score_liquidity_climax,
        score_position_size,
        score_price_acceleration,
        score_time_horizon,
        score_portfolio_concentration,
        score_event_risk,
        score_narrative_risk,
        score_execution_risk,
    )
except Exception as exc:
    st.error(f"sell_engine.py could not be imported: {exc}")
    st.stop()


st.set_page_config(page_title="SPCX Live Model", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #050505;
    }

    [data-testid="stSidebar"] {
        background: #0B0B0B;
    }

    [data-testid="stMetric"] {
        background: #111111;
        border: 1px solid #222222;
        border-radius: 12px;
        padding: 10px;
    }

    div[data-testid="stDataFrame"] {
        background: #111111;
    }

    h1,h2,h3 {
        color: #E5E5E5 !important;
    }

    p,span,label {
        color: #CFCFCF !important;
    }

    section[data-testid="stSidebar"] * {
        color: #D9D9D9 !important;
    }

    .block-container {
        padding-top: 1.4rem;
        max-width: 1440px;
    }

    .spcx-hero-card {
        margin-top: 0.75rem;
        margin-bottom: 1.25rem;
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 18px;
        background:
            radial-gradient(circle at top left, rgba(75, 150, 255, 0.20), transparent 36%),
            linear-gradient(135deg, rgba(14, 22, 39, 0.95), rgba(3, 7, 15, 0.92));
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.35);
    }

    .spcx-eyebrow {
        color: #8fc7ff;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .spcx-title {
        color: #ffffff;
        font-size: 2.35rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
    }

    .spcx-subtitle {
        color: rgba(255, 255, 255, 0.74);
        font-size: 1.02rem;
        margin-top: 0.55rem;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
HERO_IMAGE = APP_DIR / "assets" / "spacex_ipo_robotech.png"

FAIR_VALUE_TARGET = 63.0
DEFAULT_TARGETS = [63, 160, 170, 190, 210, 250, 350, 450, 500]
DEFAULT_IPO_SHARES_SOLD = 555_560_000.0
DEFAULT_IMPLIED_TOTAL_SHARES = 13_111_111_111.11

MARKET_OPEN_UTC = time(13, 30)
MARKET_CLOSE_UTC = time(20, 0)
SESSION_MINUTES = 390.0


def now_utc():
    return datetime.now(timezone.utc)


def now_text():
    return now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_utc(text):
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def money(value):
    if value is None or pd.isna(value):
        return "-"
    return f"${value:,.2f}"


def whole(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def percent(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.1%}"


def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name)


def quote_age_minutes(snapshot_time):
    parsed = parse_utc(snapshot_time)
    if parsed is None:
        return None
    return max(0.0, (now_utc() - parsed).total_seconds() / 60.0)


def elapsed_session_minutes(snapshot_time):
    parsed = parse_utc(snapshot_time)
    if parsed is None:
        return 120.0

    session_open = datetime.combine(parsed.date(), MARKET_OPEN_UTC, tzinfo=timezone.utc)
    session_close = datetime.combine(parsed.date(), MARKET_CLOSE_UTC, tzinfo=timezone.utc)

    if parsed <= session_open:
        return 1.0
    if parsed >= session_close:
        return SESSION_MINUTES
    return max(1.0, (parsed - session_open).total_seconds() / 60.0)


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clamp(value, low, high):
    return max(low, min(high, value))


def target_direction(target, price):
    if target < price:
        return "Downside"
    if target > price:
        return "Upside"
    return "At spot"


def probability_to_target(target, price, high, low, equilibrium, snapshot_time, horizon_minutes):
    if price <= 0:
        return 0.0

    if abs(target - price) < 1e-9:
        return 0.98

    distance = abs(target - price)
    daily_range = max(high - low, price * 0.005, 0.01)
    elapsed = elapsed_session_minutes(snapshot_time)

    sigma_per_sqrt_minute = daily_range / (
        math.sqrt(8.0 / math.pi) * math.sqrt(max(elapsed, 1.0))
    )
    sigma_horizon = max(
        sigma_per_sqrt_minute * math.sqrt(max(horizon_minutes, 1.0)),
        price * 0.0025,
    )

    base_probability = 2.0 * (1.0 - normal_cdf(distance / sigma_horizon))
    momentum = clamp((price - equilibrium) / daily_range, -1.0, 1.0)

    if target > price:
        base_probability *= 1.0 + 0.25 * momentum
    elif target < price:
        base_probability *= 1.0 - 0.25 * momentum

    return clamp(base_probability, 0.0, 0.98)


def probability_text(probability):
    if probability is None or pd.isna(probability):
        return "-"
    if 0 < probability < 0.001:
        return "<0.1%"
    return f"{probability:.1%}"


def practical_zone(target, price, probability):
    p = probability_text(probability)

    if abs(target - FAIR_VALUE_TARGET) < 1e-9:
        return f"FV/DD {p}"

    if target < price:
        return f"Support {p}"

    if probability >= 0.50:
        return f"Active {p}"

    if probability >= 0.15:
        return f"Watch {p}"

    if probability >= 0.03:
        return f"Stretch {p}"

    return f"Tail {p}"


def polygon_snapshot(ticker, api_key):
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"

    response = requests.get(url, params={"apiKey": api_key}, timeout=12)
    response.raise_for_status()

    payload = response.json()
    data = payload.get("ticker", {})
    day = data.get("day", {}) or {}
    last_trade = data.get("lastTrade", {}) or {}
    previous_day = data.get("prevDay", {}) or {}

    price = last_trade.get("p") or day.get("c") or previous_day.get("c")
    if price is None:
        raise RuntimeError("Polygon returned no current price")

    timestamp = last_trade.get("t")
    if timestamp:
        snapshot_time = datetime.fromtimestamp(
            timestamp / 1_000_000_000,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        snapshot_time = now_text()

    return {
        "provider": "Polygon",
        "ticker": ticker,
        "price": float(price),
        "open": float(day.get("o") or previous_day.get("o") or price),
        "high": float(day.get("h") or previous_day.get("h") or price),
        "low": float(day.get("l") or previous_day.get("l") or price),
        "volume": float(day.get("v") or previous_day.get("v") or 0),
        "snapshot_time": snapshot_time,
        "app_refresh_time": now_text(),
    }


def yahoo_snapshot(ticker):
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    ticker_obj = yf.Ticker(ticker)

    history = ticker_obj.history(
        period="1d",
        interval="1m",
        prepost=True,
    )

    if history.empty:
        history = ticker_obj.history(
            period="5d",
            interval="1d",
        )

    if history.empty:
        raise RuntimeError("Yahoo/yfinance returned no data")

    fast = ticker_obj.fast_info

    price = (
        fast.get("lastPrice")
        or fast.get("last_price")
        or float(history.iloc[-1]["Close"])
    )

    timestamp = history.index[-1]

    try:
        snapshot_time = timestamp.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        snapshot_time = now_text()

    return {
        "provider": "Yahoo/yfinance",
        "ticker": ticker,
        "price": float(price),
        "open": float(history["Open"].iloc[0]),
        "high": float(history["High"].max()),
        "low": float(history["Low"].min()),
        "volume": float(history["Volume"].sum()),
        "snapshot_time": snapshot_time,
        "app_refresh_time": now_text(),
    }


def manual_snapshot(ticker, price, high, low, open_price, volume):
    return {
        "provider": "Manual override",
        "ticker": ticker,
        "price": float(price),
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "volume": float(volume),
        "snapshot_time": now_text(),
        "app_refresh_time": now_text(),
    }


def get_snapshot(provider, ticker, manual_values):
    if provider == "Polygon":
        api_key = get_secret("POLYGON_API_KEY")

        if api_key:
            try:
                return polygon_snapshot(ticker, api_key), None
            except Exception as exc:
                return yahoo_snapshot(ticker), f"Polygon failed: {exc}. Yahoo/yfinance fallback active."

        return yahoo_snapshot(ticker), "No POLYGON_API_KEY found. Yahoo/yfinance fallback active."

    if provider == "Yahoo/yfinance":
        return yahoo_snapshot(ticker), "Yahoo/yfinance is near-live and can be delayed."

    return manual_snapshot(ticker, **manual_values), None


def build_tables(
    snapshot,
    targets,
    risk_weight,
    position_size,
    ipo_shares_sold,
    implied_total_shares,
    horizon_minutes,
):
    price = snapshot["price"]
    high = snapshot["high"]
    low = snapshot["low"]
    open_price = snapshot["open"]
    volume = snapshot["volume"]

    equilibrium = (high + low) / 2.0
    daily_range = high - low

    sale_point = equilibrium + 0.5 * (high - equilibrium) - risk_weight * (equilibrium - low)
    stop = equilibrium - risk_weight * (equilibrium - low)
    risk_to_stop = price - stop

    ladder_rows = []

    for target in targets:
        direction = target_direction(target, price)
        probability = probability_to_target(
            target,
            price,
            high,
            low,
            equilibrium,
            snapshot["snapshot_time"],
            horizon_minutes,
        )

        delta = target - price
        reward = max(delta, 0.0)
        downside_move = max(price - target, 0.0)
        reward_risk = reward / risk_to_stop if risk_to_stop else 0.0
        expected_value = probability * reward - (1.0 - probability) * max(risk_to_stop, 0.0)

        if direction == "Downside":
            action = "DOWNSIDE WATCH"
        elif direction == "At spot":
            action = "AT SPOT"
        else:
            action = "PLACE LIMIT SELL"

        zone = practical_zone(target, price, probability)

        ladder_rows.append(
            {
                "Target sell price": target,
                "Direction": direction,
                "Practical zone": zone,
                "Action": action,
                "Current price": price,
                "Delta to target $/sh": delta,
                "Delta to target %": delta / price if price else None,
                "Reward $/sh": reward,
                "Downside move $/sh": downside_move,
                "Stop / invalidation": stop,
                "Risk to stop $/sh": risk_to_stop,
                "Risk to stop %": risk_to_stop / price if price else None,
                "Reward / risk": reward_risk,
                "Probability to target": probability,
                "Probability label": zone,
                "Expected value $/sh": expected_value,
                "Position size": position_size,
                "Estimated P/L at target": delta * position_size,
            }
        )

    ladder_df = pd.DataFrame(ladder_rows)

    model_df = pd.DataFrame(
        [
            ["Risk floor", low],
            ["Equilibrium", equilibrium],
            ["Live price", price],
            ["Optimistic ceiling", high],
            ["Daily range", daily_range],
            ["Range % of live price", daily_range / price if price else None],
            ["Upside to high", high - price],
            ["Downside to floor", price - low],
            ["Two-hour sale point", sale_point],
            ["Delta from live price", sale_point - price],
            ["Potential gain to model sale point", (sale_point - price) / price if price else None],
            ["Estimated P/L at sale point", (sale_point - price) * position_size],
            ["Stop / invalidation", stop],
            ["Risk to stop", risk_to_stop / price if price else None],
            ["Live vs open", price / open_price - 1 if open_price else None],
            ["Probability horizon minutes", horizon_minutes],
        ],
        columns=["Metric", "Value"],
    )

    shares_held = implied_total_shares - ipo_shares_sold

    stats_df = pd.DataFrame(
        [
            ["Current price", price],
            ["Intraday high", high],
            ["Intraday low", low],
            ["Open", open_price],
            ["Intraday volume", volume],
            ["Approx. dollar volume", price * volume],
            ["IPO shares sold", ipo_shares_sold],
            ["Implied total shares outstanding", implied_total_shares],
            ["Shares still held / not sold", shares_held],
            ["Free float sold in IPO", ipo_shares_sold / implied_total_shares if implied_total_shares else None],
            ["Still held / not sold", shares_held / implied_total_shares if implied_total_shares else None],
            ["Turnover vs IPO shares sold", volume / ipo_shares_sold if ipo_shares_sold else None],
            ["Turnover vs implied total shares", volume / implied_total_shares if implied_total_shares else None],
            ["Live implied market cap", price * implied_total_shares],
            ["Quote timestamp", snapshot["snapshot_time"]],
            ["App refresh timestamp", snapshot["app_refresh_time"]],
            ["Quote age minutes", quote_age_minutes(snapshot["snapshot_time"])],
            ["Probability horizon minutes", horizon_minutes],
            ["Fair value target", FAIR_VALUE_TARGET],
            ["Probability model", "Realized-range first-passage heuristic, not option-implied probability"],
            ["Provider", snapshot["provider"]],
        ],
        columns=["Metric", "Value"],
    )

    return model_df, ladder_df, stats_df


def format_model_value(metric, value):
    if isinstance(value, str):
        return value

    metric_lower = metric.lower()

    if (
        "range %" in metric_lower
        or "gain" in metric_lower
        or "risk to stop" in metric_lower
        or "live vs open" in metric_lower
    ):
        return percent(value)

    if "minutes" in metric_lower:
        return f"{value:,.1f}"

    if any(
        word in metric_lower
        for word in [
            "floor",
            "equilibrium",
            "price",
            "ceiling",
            "range",
            "upside",
            "downside",
            "point",
            "delta",
            "p/l",
            "stop",
        ]
    ):
        return money(value)

    return f"{value:,.3f}"


def format_stat_value(metric, value):
    if isinstance(value, str):
        return value

    metric_lower = metric.lower()

    if (
        "price" in metric_lower
        or "high" in metric_lower
        or "low" in metric_lower
        or metric_lower == "open"
        or "dollar volume" in metric_lower
        or "market cap" in metric_lower
    ):
        return money(value)

    if "shares" in metric_lower or "volume" in metric_lower:
        return whole(value)

    if "float" in metric_lower or "held" in metric_lower or "turnover" in metric_lower:
        return percent(value)

    if "age" in metric_lower or "minutes" in metric_lower:
        return f"{value:,.1f}"

    return f"{value:,.2f}"


def make_excel(snapshot, model_df, ladder_df, stats_df, sell_engine_df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([snapshot]).to_excel(writer, index=False, sheet_name="Inputs")
        model_df.to_excel(writer, index=False, sheet_name="Model")
        ladder_df.to_excel(writer, index=False, sheet_name="Market Sales")
        stats_df.to_excel(writer, index=False, sheet_name="Market Stats")
        sell_engine_df.to_excel(writer, index=False, sheet_name="Sell Engine")

    return output.getvalue()


def recommendation_from_score(score):
    if score < 13:
        return "HOLD"
    if score < 21:
        return "PREPARE LIMIT ORDERS"
    if score < 34:
        return "SELL 13%"
    if score < 55:
        return "SELL 21%"
    if score < 89:
        return "SELL 34%"
    return "SELL 55%"


def action_palette(action):
    if action == "HOLD":
        return "#1f9d55", "#071a10"
    if action == "PREPARE LIMIT ORDERS":
        return "#d6a600", "#1d1703"
    if action == "SELL 13%":
        return "#d9822b", "#1f1004"
    if action == "SELL 21%":
        return "#f97016", "#241004"
    if action == "SELL 34%":
        return "#ef4444", "#230707"
    if action == "SELL 55%":
        return "#dc2626", "#260606"
    return "#9ca3af", "#111111"


def render_action_banner(action, score):
    accent, background = action_palette(action)
    st.markdown(
        f"""
        <div style="
            margin-top: 0.35rem;
            margin-bottom: 1.1rem;
            padding: 1.05rem 1.25rem;
            border-radius: 16px;
            border: 1px solid {accent};
            background: linear-gradient(135deg, {background}, #090909);
            box-shadow: 0 0 28px rgba(0,0,0,0.38);
        ">
            <div style="
                color: #A9A9A9;
                font-size: 0.82rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            ">
                Recommended action
            </div>
            <div style="
                color: {accent};
                font-size: 3.1rem;
                line-height: 1.0;
                font-weight: 800;
                white-space: nowrap;
            ">
                {action}
            </div>
            <div style="
                color: #D1D5DB;
                font-size: 0.95rem;
                margin-top: 0.55rem;
            ">
                Current Engine Score: {score:.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_expected_upside_pct(current_price, targets, probabilities):
    if current_price <= 0:
        return 0.0
    ev = 0.0
    for target, probability in zip(targets, probabilities):
        if target > current_price:
            ev += probability * (target - current_price)
    return ev / current_price * 100.0


def build_sell_engine_table(
    current_price,
    fair_value,
    sell_targets,
    sell_probabilities,
    price_5d,
    volatility_ratio,
    current_volume,
    avg_volume_20d,
    day_low,
    day_high,
    position_value,
    total_portfolio_value,
    price_return_1d,
    price_return_3d,
    price_return_5d,
    days_until_money_needed,
    second_largest_position,
    event_severity,
    event_probability,
    days_to_event,
    narrative_risk_manual,
    starship_status_risk,
    launch_cadence_risk,
    nasa_dependence_risk,
    defense_contracts_risk,
):
    valuation_score = score_valuation(current_price, fair_value)
    upside_score = score_expected_upside(current_price, sell_targets, sell_probabilities)
    momentum_score = score_momentum(current_price, price_5d)
    acceleration_score = score_price_acceleration(price_return_1d, price_return_3d, price_return_5d)
    volatility_score = score_volatility_ratio(volatility_ratio)
    liquidity_score = score_liquidity_climax(current_price, day_low, day_high, current_volume, avg_volume_20d)
    position_score = score_position_size(position_value, total_portfolio_value)
    time_score = score_time_horizon(days_until_money_needed)
    concentration_score = score_portfolio_concentration(position_value, total_portfolio_value, second_largest_position)
    event_score = score_event_risk(event_severity, event_probability, days_to_event)
    narrative_score = score_narrative_risk(manual_score=narrative_risk_manual)
    execution_score = score_execution_risk(
        starship_status_risk,
        launch_cadence_risk,
        nasa_dependence_risk,
        defense_contracts_risk,
    )

    valuation_raw = ((current_price - fair_value) / fair_value * 100.0) if fair_value > 0 else 0.0
    expected_upside_raw = compute_expected_upside_pct(current_price, sell_targets, sell_probabilities)
    momentum_raw = ((current_price - price_5d) / price_5d * 100.0) if price_5d > 0 else 0.0
    volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0.0
    price_position = ((current_price - day_low) / (day_high - day_low)) if day_high > day_low else 0.0
    position_pct = position_value / total_portfolio_value * 100.0 if total_portfolio_value > 0 else 0.0
    concentration_ratio = position_value / second_largest_position if second_largest_position > 0 else 0.0

    rows = [
        {"Block": "Market", "Factor": "Valuation", "Raw Value": f"{valuation_raw:.1f}% premium to fair value", "Score": valuation_score, "Weight": 10.0},
        {"Block": "Market", "Factor": "Expected Upside", "Raw Value": f"{expected_upside_raw:.1f}% heuristic EV", "Score": upside_score, "Weight": 9.0},
        {"Block": "Market", "Factor": "Momentum", "Raw Value": f"{momentum_raw:.1f}% vs reference", "Score": momentum_score, "Weight": 8.0},
        {"Block": "Market", "Factor": "Price Acceleration", "Raw Value": f"1d {price_return_1d:.1f}%, 3d {price_return_3d:.1f}%, 5d {price_return_5d:.1f}%", "Score": acceleration_score, "Weight": 4.0},
        {"Block": "Market", "Factor": "Volatility", "Raw Value": f"{volatility_ratio:.2f}x", "Score": volatility_score, "Weight": 5.0},
        {"Block": "Market", "Factor": "Liquidity Climax", "Raw Value": f"{volume_ratio:.2f}x volume, {price_position:.1%} of day range", "Score": liquidity_score, "Weight": 5.0},
        {"Block": "Portfolio", "Factor": "Position Size", "Raw Value": f"{position_pct:.1f}% of portfolio", "Score": position_score, "Weight": 9.0},
        {"Block": "Portfolio", "Factor": "Time Horizon", "Raw Value": f"{days_until_money_needed:.0f} days until money needed", "Score": time_score, "Weight": 10.0},
        {"Block": "Portfolio", "Factor": "Portfolio Concentration", "Raw Value": f"{position_pct:.1f}% portfolio, {concentration_ratio:.2f}x second position", "Score": concentration_score, "Weight": 13.0},
        {"Block": "Business / Judgment", "Factor": "Event Risk", "Raw Value": f"severity {event_severity:.1f}, p {event_probability:.2f}, {days_to_event:.0f} days", "Score": event_score, "Weight": 9.0},
        {"Block": "Business / Judgment", "Factor": "Narrative Risk", "Raw Value": f"{narrative_risk_manual:.0f} manual score", "Score": narrative_score, "Weight": 8.0},
        {"Block": "Business / Judgment", "Factor": "Execution Risk", "Raw Value": f"Starship {starship_status_risk:.0f}, cadence {launch_cadence_risk:.0f}, NASA {nasa_dependence_risk:.0f}, defense {defense_contracts_risk:.0f}", "Score": execution_score, "Weight": 10.0},
    ]

    sell_engine_df = pd.DataFrame(rows)
    sell_engine_df["Contribution"] = sell_engine_df["Score"] * sell_engine_df["Weight"] / 100.0

    implemented_weight = float(sell_engine_df["Weight"].sum())
    implemented_contribution = float(sell_engine_df["Contribution"].sum())
    current_engine_score = implemented_contribution / implemented_weight * 100.0 if implemented_weight else 0.0

    block_scores = {}
    for block, block_df in sell_engine_df.groupby("Block"):
        block_weight = float(block_df["Weight"].sum())
        block_contribution = float(block_df["Contribution"].sum())
        block_scores[block] = block_contribution / block_weight * 100.0 if block_weight else 0.0

    return sell_engine_df, implemented_contribution, implemented_weight, current_engine_score, block_scores

if HERO_IMAGE.exists():
    st.image(str(HERO_IMAGE), use_container_width=True)

st.markdown(
    """
    <div style="text-align:center; margin-top:-30px; margin-bottom:25px;">
    <h1 style="font-size:42px; font-weight:700; color:#E5E5E5; margin-bottom:5px;">
    SPCX LIVE TRADING MODEL
    </h1>
    <div style="font-size:18px; color:#A0A0A0;">
    Real-Time Probability - Risk / Reward - Float Analysis
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="spcx-hero-card">
        <div class="spcx-eyebrow">SPCX live IPO execution dashboard</div>
        <h1 class="spcx-title">SPCX Live Sale Model</h1>
        <p class="spcx-subtitle">Near-live quote, $63 fair-value drawdown reference, sell-target probabilities, P/L ladder, float math, and Sell Decision Engine V1.2.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")

    ticker = st.text_input("Ticker", value="SPCX").upper().strip()

    default_provider_index = 0 if get_secret("POLYGON_API_KEY") else 1
    provider = st.selectbox(
        "Quote provider",
        ["Polygon", "Yahoo/yfinance", "Manual override"],
        index=default_provider_index,
    )

    refresh_seconds = st.number_input(
        "Auto-refresh seconds",
        min_value=10,
        max_value=600,
        value=30,
        step=10,
    )

    auto_refresh = st.toggle("Auto-refresh", value=True)

    if st.button("Refresh now"):
        st.rerun()

    risk_weight = st.number_input(
        "Risk weight",
        min_value=0.0,
        max_value=2.0,
        value=0.35,
        step=0.05,
    )

    horizon_minutes = st.number_input(
        "Probability horizon minutes",
        min_value=5.0,
        max_value=390.0,
        value=120.0,
        step=5.0,
    )

    position_size = st.number_input(
        "Position size",
        min_value=1.0,
        value=100.0,
        step=1.0,
    )

    targets_text = st.text_input(
        "Price reference points",
        value=", ".join(str(x) for x in DEFAULT_TARGETS),
    )

    ipo_shares_sold = st.number_input(
        "IPO shares sold",
        min_value=1.0,
        value=float(DEFAULT_IPO_SHARES_SOLD),
        step=1000.0,
    )

    implied_total_shares = st.number_input(
        "Implied total shares",
        min_value=1.0,
        value=float(DEFAULT_IMPLIED_TOTAL_SHARES),
        step=1000.0,
    )

    st.subheader("Manual values")

    manual_values = {
        "price": st.number_input("Manual price", min_value=0.0, value=176.38, step=0.01),
        "high": st.number_input("Manual high", min_value=0.0, value=176.38, step=0.01),
        "low": st.number_input("Manual low", min_value=0.0, value=150.20, step=0.01),
        "open_price": st.number_input("Manual open", min_value=0.0, value=150.00, step=0.01),
        "volume": st.number_input("Manual volume", min_value=0.0, value=315_425_119.0, step=1000.0),
    }

    st.subheader("Sell Engine inputs")

    fair_value_input = st.number_input(
        "Fair value for Sell Engine",
        min_value=0.01,
        value=float(FAIR_VALUE_TARGET),
        step=1.0,
    )

    price_5d_input = st.number_input(
        "Reference price for momentum",
        min_value=0.01,
        value=135.00,
        step=0.01,
    )

    volatility_ratio_input = st.number_input(
        "Volatility ratio",
        min_value=0.01,
        value=1.00,
        step=0.05,
    )

    avg_volume_20d_input = st.number_input(
        "Average volume 20d",
        min_value=1.0,
        value=315_425_119.0,
        step=1000.0,
    )

    total_portfolio_value_input = st.number_input(
        "Total portfolio value",
        min_value=1.0,
        value=100_000.0,
        step=1000.0,
    )

    price_return_1d_input = st.number_input("Price return 1d %", value=12.0, step=0.5)
    price_return_3d_input = st.number_input("Price return 3d %", value=28.0, step=0.5)
    price_return_5d_input = st.number_input("Price return 5d %", value=58.0, step=0.5)

    days_until_money_needed_input = st.number_input(
        "Days until money needed",
        min_value=0.0,
        value=90.0,
        step=1.0,
    )

    second_largest_position_input = st.number_input(
        "Second largest position value",
        min_value=0.0,
        value=10_000.0,
        step=1000.0,
    )

    st.subheader("Business / Judgment inputs")

    event_severity_input = st.slider("Event severity", 0.0, 10.0, 5.0, 0.5)
    event_probability_input = st.slider("Event probability", 0.0, 1.0, 0.35, 0.05)
    days_to_event_input = st.number_input("Days to event", min_value=0.0, value=14.0, step=1.0)

    narrative_risk_input = st.slider("Narrative risk manual score", 0.0, 100.0, 50.0, 1.0)

    starship_status_risk_input = st.slider("Starship status risk", 0.0, 100.0, 50.0, 1.0)
    launch_cadence_risk_input = st.slider("Launch cadence risk", 0.0, 100.0, 40.0, 1.0)
    nasa_dependence_risk_input = st.slider("NASA dependence risk", 0.0, 100.0, 35.0, 1.0)
    defense_contracts_risk_input = st.slider("Defense contracts risk", 0.0, 100.0, 45.0, 1.0)

refresh_count = None

if auto_refresh:
    refresh_count = st_autorefresh(
        interval=int(refresh_seconds * 1000),
        key="spcx_live_refresh",
    )

try:
    targets = [float(item.strip()) for item in targets_text.split(",") if item.strip()]
    snapshot, warning = get_snapshot(provider, ticker, manual_values)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if warning:
    st.info(warning)

model_df, ladder_df, stats_df = build_tables(
    snapshot=snapshot,
    targets=targets,
    risk_weight=risk_weight,
    position_size=position_size,
    ipo_shares_sold=ipo_shares_sold,
    implied_total_shares=implied_total_shares,
    horizon_minutes=horizon_minutes,
)

quote_age = quote_age_minutes(snapshot["snapshot_time"])

cols = st.columns(6)
cols[0].metric("Current price", money(snapshot["price"]))
cols[1].metric("High", money(snapshot["high"]))
cols[2].metric("Low", money(snapshot["low"]))
cols[3].metric("Volume", whole(snapshot["volume"]))
cols[4].metric("Quote age", "-" if quote_age is None else f"{quote_age:.1f} min")
cols[5].metric("Provider", snapshot["provider"])

st.caption(
    f"Quote timestamp: {snapshot['snapshot_time']} | App refreshed: {snapshot['app_refresh_time']}"
    + (f" | Auto-refresh run: {refresh_count}" if refresh_count is not None else "")
)

left, right = st.columns([1.15, 1.0])

with left:
    st.subheader("Practical execution table")

    st.dataframe(
        ladder_df.drop(columns=["Probability label"]).style.format(
            {
                "Target sell price": "${:,.2f}",
                "Current price": "${:,.2f}",
                "Delta to target $/sh": "${:,.2f}",
                "Delta to target %": "{:.2%}",
                "Reward $/sh": "${:,.2f}",
                "Downside move $/sh": "${:,.2f}",
                "Stop / invalidation": "${:,.2f}",
                "Risk to stop $/sh": "${:,.2f}",
                "Risk to stop %": "{:.2%}",
                "Reward / risk": "{:.2f}",
                "Probability to target": "{:.1%}",
                "Expected value $/sh": "${:,.2f}",
                "Position size": "{:,.0f}",
                "Estimated P/L at target": "${:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Model")

    model_display = model_df.copy()
    model_display["Value"] = [
        format_model_value(metric, value)
        for metric, value in zip(model_display["Metric"], model_display["Value"])
    ]

    st.dataframe(model_display, use_container_width=True, hide_index=True)

sell_ladder_df = ladder_df[ladder_df["Target sell price"] > snapshot["price"]].copy()
fair_value_df = ladder_df[ladder_df["Target sell price"] == FAIR_VALUE_TARGET].copy()

sell_engine_targets = sell_ladder_df["Target sell price"].astype(float).tolist()
sell_engine_probabilities = sell_ladder_df["Probability to target"].astype(float).tolist()
position_value = float(snapshot["price"]) * float(position_size)

sell_engine_df, sell_engine_partial, sell_engine_weight, sell_engine_score, block_scores = build_sell_engine_table(
    current_price=float(snapshot["price"]),
    fair_value=float(fair_value_input),
    sell_targets=sell_engine_targets,
    sell_probabilities=sell_engine_probabilities,
    price_5d=float(price_5d_input),
    volatility_ratio=float(volatility_ratio_input),
    current_volume=float(snapshot["volume"]),
    avg_volume_20d=float(avg_volume_20d_input),
    day_low=float(snapshot["low"]),
    day_high=float(snapshot["high"]),
    position_value=float(position_value),
    total_portfolio_value=float(total_portfolio_value_input),
    price_return_1d=float(price_return_1d_input),
    price_return_3d=float(price_return_3d_input),
    price_return_5d=float(price_return_5d_input),
    days_until_money_needed=float(days_until_money_needed_input),
    second_largest_position=float(second_largest_position_input),
    event_severity=float(event_severity_input),
    event_probability=float(event_probability_input),
    days_to_event=float(days_to_event_input),
    narrative_risk_manual=float(narrative_risk_input),
    starship_status_risk=float(starship_status_risk_input),
    launch_cadence_risk=float(launch_cadence_risk_input),
    nasa_dependence_risk=float(nasa_dependence_risk_input),
    defense_contracts_risk=float(defense_contracts_risk_input),
)

st.subheader("SELL DECISION ENGINE V1.0")

action = recommendation_from_score(sell_engine_score)

engine_cols = st.columns(5)
engine_cols[0].metric("Implemented Weight", f"{sell_engine_weight:.0f}%")
engine_cols[1].metric("Total Contribution", f"{sell_engine_partial:.1f} / {sell_engine_weight:.0f}")
engine_cols[2].metric("Current Engine Score", f"{sell_engine_score:.1f}")
engine_cols[3].metric("Market Score", f"{block_scores.get('Market', 0.0):.1f}")
engine_cols[4].metric("Portfolio Score", f"{block_scores.get('Portfolio', 0.0):.1f}")

render_action_banner(action, sell_engine_score)

business_cols = st.columns(1)
business_cols[0].metric("Business / Judgment Score", f"{block_scores.get('Business / Judgment', 0.0):.1f}")

st.dataframe(
    sell_engine_df.style.format(
        {
            "Score": "{:.0f}",
            "Weight": "{:.1f}",
            "Contribution": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Sell Engine V1.0 uses all 12 factors with 100% total model weight. Expected Upside is a heuristic based on target reach probabilities, not option-implied EV."
)

if not fair_value_df.empty:
    fair_value_row = fair_value_df.iloc[0]

    st.subheader("Fair value / drawdown reference")

    fv_cols = st.columns(3)

    fv_cols[0].metric(
        "Fair value reference",
        money(FAIR_VALUE_TARGET),
        f"{fair_value_row['Delta to target %']:.1%} from current",
    )

    fv_cols[1].metric(
        "Downside probability",
        probability_text(fair_value_row["Probability to target"]),
    )

    fv_cols[2].metric(
        "Downside move / share",
        money(fair_value_row["Downside move $/sh"]),
    )

st.subheader("Probability by sell target")

prob_chart = alt.Chart(sell_ladder_df).mark_bar().encode(
    x=alt.X("Target sell price:O", title="Sell target"),
    y=alt.Y("Probability to target:Q", title="Probability", axis=alt.Axis(format="%")),
    tooltip=[
        alt.Tooltip("Target sell price:Q", title="Target", format="$.2f"),
        alt.Tooltip("Practical zone:N", title="Zone"),
        alt.Tooltip("Probability to target:Q", title="Probability", format=".1%"),
        alt.Tooltip("Reward / risk:Q", title="Reward / risk", format=".2f"),
        alt.Tooltip("Expected value $/sh:Q", title="EV $/sh", format="$.2f"),
    ],
).properties(height=320)

prob_text = alt.Chart(sell_ladder_df).mark_text(
    dy=-8,
    fontSize=13,
    fontWeight="bold",
    color="white",
).encode(
    x=alt.X("Target sell price:O"),
    y=alt.Y("Probability to target:Q"),
    text=alt.Text("Probability label:N"),
)

st.altair_chart(prob_chart + prob_text, use_container_width=True)

st.subheader("P/L scenario by sell target")

pl_chart = alt.Chart(sell_ladder_df).mark_line(point=True).encode(
    x=alt.X("Target sell price:Q", title="Sell target"),
    y=alt.Y("Estimated P/L at target:Q", title="Estimated P/L"),
    tooltip=[
        alt.Tooltip("Target sell price:Q", title="Target", format="$.2f"),
        alt.Tooltip("Estimated P/L at target:Q", title="Estimated P/L", format="$.2f"),
        alt.Tooltip("Reward / risk:Q", title="Reward / risk", format=".2f"),
        alt.Tooltip("Probability to target:Q", title="Probability", format=".1%"),
    ],
).properties(height=320)

st.altair_chart(pl_chart, use_container_width=True)

st.subheader("Risk vs reward by target")

risk_reward_df = sell_ladder_df.melt(
    id_vars=["Target sell price"],
    value_vars=["Reward $/sh", "Risk to stop $/sh"],
    var_name="Line",
    value_name="USD per share",
)

risk_reward_chart = alt.Chart(risk_reward_df).mark_line(point=True).encode(
    x=alt.X("Target sell price:Q", title="Sell target"),
    y=alt.Y("USD per share:Q", title="USD per share"),
    color=alt.Color("Line:N", title="Line"),
    tooltip=[
        alt.Tooltip("Target sell price:Q", title="Target", format="$.2f"),
        alt.Tooltip("Line:N", title="Line"),
        alt.Tooltip("USD per share:Q", title="USD per share", format="$.2f"),
    ],
).properties(height=320)

st.altair_chart(risk_reward_chart, use_container_width=True)

st.caption(
    "Probability is a realized-range first-passage heuristic based on current quote, intraday high/low, elapsed session time, selected horizon, and momentum position. It is not option-implied probability and not a guarantee."
)

st.subheader("Market stats and float math")

stats_display = stats_df.copy()
stats_display["Value"] = [
    format_stat_value(metric, value)
    for metric, value in zip(stats_display["Metric"], stats_display["Value"])
]

st.dataframe(stats_display, use_container_width=True, hide_index=True)

excel_bytes = make_excel(snapshot, model_df, ladder_df, stats_df, sell_engine_df)

st.download_button(
    "Download updated Excel model",
    data=excel_bytes,
    file_name=f"{ticker.lower()}_live_model.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
