import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="SPCX Live Model", layout="wide")

DEFAULT_TARGETS = [160, 170, 190, 210, 250, 350, 450, 500]
DEFAULT_IPO_SHARES_SOLD = 555_560_000
DEFAULT_IMPLIED_TOTAL_SHARES = 13_111_111_111.11


def money(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"${x:,.2f}"


def num(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.0f}"


def get_secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_string() -> str:
    return now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_utc_string(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def quote_age_minutes(snapshot_time: str) -> Optional[float]:
    parsed = parse_utc_string(snapshot_time)
    if parsed is None:
        return None
    return max(0.0, (now_utc() - parsed).total_seconds() / 60.0)


def polygon_snapshot(ticker: str, api_key: str) -> Dict:
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
    r = requests.get(url, params={"apiKey": api_key}, timeout=12)
    r.raise_for_status()
    payload = r.json()
    t = payload.get("ticker", {})
    day = t.get("day", {}) or {}
    last_trade = t.get("lastTrade", {}) or {}
    prev_day = t.get("prevDay", {}) or {}

    price = last_trade.get("p") or day.get("c") or prev_day.get("c")
    if price is None:
        raise ValueError("Polygon response did not include a current price")

    ts_ns = last_trade.get("t")
    if ts_ns:
        snapshot_time = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        snapshot_time = now_utc_string()

    return {
        "provider": "Polygon snapshot",
        "ticker": ticker.upper(),
        "price": float(price),
        "open": float(day.get("o") or prev_day.get("o") or price),
        "high": float(day.get("h") or prev_day.get("h") or price),
        "low": float(day.get("l") or prev_day.get("l") or price),
        "volume": float(day.get("v") or prev_day.get("v") or 0),
        "snapshot_time": snapshot_time,
        "app_refresh_time": now_utc_string(),
    }


def yahoo_intraday(ticker: str) -> Dict:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    hist = yf.Ticker(ticker.upper()).history(period="1d", interval="1m", prepost=False)
    if hist.empty:
        hist = yf.Ticker(ticker.upper()).history(period="5d", interval="1d")
    if hist.empty:
        raise ValueError("Yahoo/yfinance returned no price history")

    last = hist.iloc[-1]
    snapshot_time = hist.index[-1]
    try:
        snapshot_time = snapshot_time.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        snapshot_time = now_utc_string()

    return {
        "provider": "Yahoo/yfinance fallback",
        "ticker": ticker.upper(),
        "price": float(last["Close"]),
        "open": float(hist["Open"].iloc[0]),
        "high": float(hist["High"].max()),
        "low": float(hist["Low"].min()),
        "volume": float(hist["Volume"].sum()),
        "snapshot_time": snapshot_time,
        "app_refresh_time": now_utc_string(),
    }


def manual_snapshot(ticker: str, price: float, high: float, low: float, open_price: float, volume: float) -> Dict:
    return {
        "provider": "Manual override",
        "ticker": ticker.upper(),
        "price": float(price),
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "volume": float(volume),
        "snapshot_time": now_utc_string(),
        "app_refresh_time": now_utc_string(),
    }


def get_market_snapshot(provider: str, ticker: str, manual_values: Dict[str, float]) -> Tuple[Dict, Optional[str]]:
    if provider == "Polygon":
        key = get_secret("POLYGON_API_KEY")
        if not key:
            return yahoo_intraday(ticker), "No POLYGON_API_KEY found. Yahoo/yfinance fallback is active. This may be delayed and should be treated as near-live, not exchange-direct live."
        try:
            return polygon_snapshot(ticker, key), None
        except Exception as e:
            try:
                return yahoo_intraday(ticker), f"Polygon failed: {e}. Yahoo/yfinance fallback is active."
            except Exception as e2:
                raise RuntimeError(f"Polygon failed: {e}. Yahoo fallback also failed: {e2}")
    if provider == "Yahoo/yfinance":
        return yahoo_intraday(ticker), None
    return manual_snapshot(ticker, **manual_values), None


def build_model(snapshot: Dict, targets, risk_weight: float, position_size: float, ipo_shares_sold: float, implied_total_shares: float):
    price = snapshot["price"]
    high = snapshot["high"]
    low = snapshot["low"]
    open_price = snapshot["open"]
    volume = snapshot["volume"]

    equilibrium = (high + low) / 2
    daily_range = high - low
    sale_point = equilibrium + 0.5 * (high - equilibrium) - risk_weight * (equilibrium - low)
    stop = equilibrium - risk_weight * (equilibrium - low)
    delta_from_live = sale_point - price
    risk_to_stop = price - stop

    model_rows = [
        ["Risk floor", low],
        ["Equilibrium", equilibrium],
        ["Live price", price],
        ["Optimistic ceiling", high],
        ["Daily range", daily_range],
        ["Range % of live price", daily_range / price if price else None],
        ["Upside to high", high - price],
        ["Downside to floor", price - low],
        ["Two-hour sale point", sale_point],
        ["Delta from live price", delta_from_live],
        ["Potential gain to model sale point", delta_from_live / price if price else None],
        ["Estimated P/L at sale point", delta_from_live * position_size],
        ["Stop / invalidation", stop],
        ["Risk to stop", risk_to_stop / price if price else None],
        ["Live vs open", price / open_price - 1 if open_price else None],
    ]
    model_df = pd.DataFrame(model_rows, columns=["Metric", "Value"])

    ladder_rows = []
    for target in targets:
        delta = target - price
        rr = delta / risk_to_stop if risk_to_stop else None
        ladder_rows.append({
            "Target sell price": target,
            "Action": "SELL MKT NOW" if target <= price else "PLACE LIMIT SELL",
            "Current price": price,
            "Delta to target $/sh": delta,
            "Delta to target %": delta / price if price else None,
            "Stop / invalidation": stop,
            "Risk to stop $/sh": risk_to_stop,
            "Risk to stop %": risk_to_stop / price if price else None,
            "Reward / risk": rr,
            "Position size": position_size,
            "Estimated P/L at target": delta * position_size,
        })
    ladder_df = pd.DataFrame(ladder_rows)

    shares_held = implied_total_shares - ipo_shares_sold
    age = quote_age_minutes(snapshot["snapshot_time"])
    stats_rows = [
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
        ["Quote age minutes", age],
        ["Provider", snapshot["provider"]],
    ]
    stats_df = pd.DataFrame(stats_rows, columns=["Metric", "Value"])

    return model_df, ladder_df, stats_df


def make_excel(snapshot: Dict, model_df: pd.DataFrame, ladder_df: pd.DataFrame, stats_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([snapshot]).to_excel(writer, index=False, sheet_name="Inputs")
        model_df.to_excel(writer, index=False, sheet_name="Model")
        ladder_df.to_excel(writer, index=False, sheet_name="Market Sales")
        stats_df.to_excel(writer, index=False, sheet_name="Market Stats")
    return output.getvalue()


st.title("SPCX Live Sale Model")

has_polygon_key = bool(get_secret("POLYGON_API_KEY"))
provider_options = ["Polygon", "Yahoo/yfinance", "Manual override"]
default_provider_index = 0 if has_polygon_key else 1

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="SPCX").upper().strip()
    provider = st.selectbox("Quote provider", provider_options, index=default_provider_index)
    refresh_seconds = st.number_input("Auto-refresh seconds", min_value=10, max_value=600, value=30, step=10)
    auto_refresh = st.toggle("Auto-refresh", value=True)
    if st.button("Refresh now"):
        st.rerun()
    risk_weight = st.number_input("Risk weight", min_value=0.0, max_value=2.0, value=0.35, step=0.05)
    position_size = st.number_input("Position size", min_value=1.0, value=100.0, step=1.0)
    targets_text = st.text_input("Sale targets", value=", ".join(str(x) for x in DEFAULT_TARGETS))
    ipo_shares_sold = st.number_input("IPO shares sold", min_value=1.0, value=float(DEFAULT_IPO_SHARES_SOLD), step=1000.0)
    implied_total_shares = st.number_input("Implied total shares", min_value=1.0, value=float(DEFAULT_IMPLIED_TOTAL_SHARES), step=1000.0)

    st.subheader("Manual values")
    manual_price = st.number_input("Manual price", min_value=0.0, value=176.38, step=0.01)
    manual_high = st.number_input("Manual high", min_value=0.0, value=176.38, step=0.01)
    manual_low = st.number_input("Manual low", min_value=0.0, value=150.20, step=0.01)
    manual_open = st.number_input("Manual open", min_value=0.0, value=150.00, step=0.01)
    manual_volume = st.number_input("Manual volume", min_value=0.0, value=315_425_119.0, step=1000.0)

refresh_count = None
if auto_refresh:
    refresh_count = st_autorefresh(interval=int(refresh_seconds * 1000), key="spcx_live_refresh")

try:
    targets = [float(x.strip()) for x in targets_text.split(",") if x.strip()]
except Exception:
    st.error("Sale targets must be comma-separated numbers")
    st.stop()

manual_values = {
    "price": manual_price,
    "high": manual_high,
    "low": manual_low,
    "open_price": manual_open,
    "volume": manual_volume,
}

try:
    snapshot, warning = get_market_snapshot(provider, ticker, manual_values)
except Exception as e:
    st.error(str(e))
    st.stop()

if warning:
    st.info(warning)
elif provider == "Yahoo/yfinance":
    st.info("Yahoo/yfinance fallback is active. The app refreshes every configured interval, but quote data can still be delayed by the provider.")

model_df, ladder_df, stats_df = build_model(snapshot, targets, risk_weight, position_size, ipo_shares_sold, implied_total_shares)
quote_age = quote_age_minutes(snapshot["snapshot_time"])

kpi_cols = st.columns(6)
kpi_cols[0].metric("Current price", money(snapshot["price"]))
kpi_cols[1].metric("High", money(snapshot["high"]))
kpi_cols[2].metric("Low", money(snapshot["low"]))
kpi_cols[3].metric("Volume", num(snapshot["volume"]))
kpi_cols[4].metric("Quote age", "-" if quote_age is None else f"{quote_age:.1f} min")
kpi_cols[5].metric("Provider", snapshot["provider"])

st.caption(
    f"Quote timestamp: {snapshot['snapshot_time']} | App refreshed: {snapshot['app_refresh_time']}"
    + (f" | Auto-refresh run: {refresh_count}" if refresh_count is not None else "")
)

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Sale ladder")
    st.dataframe(
        ladder_df.style.format({
            "Target sell price": "${:,.2f}",
            "Current price": "${:,.2f}",
            "Delta to target $/sh": "${:,.2f}",
            "Delta to target %": "{:.2%}",
            "Stop / invalidation": "${:,.2f}",
            "Risk to stop $/sh": "${:,.2f}",
            "Risk to stop %": "{:.2%}",
            "Reward / risk": "{:.2f}",
            "Position size": "{:,.0f}",
            "Estimated P/L at target": "${:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Model")
    st.dataframe(model_df, use_container_width=True, hide_index=True)

st.subheader("Scenario chart")
chart = alt.Chart(ladder_df).mark_line(point=True).encode(
    x=alt.X("Target sell price:Q", title="Target sell price"),
    y=alt.Y("Estimated P/L at target:Q", title="Estimated P/L"),
    tooltip=["Target sell price", "Estimated P/L at target", "Reward / risk", "Action"],
).properties(height=360)
st.altair_chart(chart, use_container_width=True)

st.subheader("Market stats and float math")
st.dataframe(stats_df, use_container_width=True, hide_index=True)

excel_bytes = make_excel(snapshot, model_df, ladder_df, stats_df)
st.download_button(
    "Download updated Excel model",
    data=excel_bytes,
    file_name=f"{ticker.lower()}_live_model.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
