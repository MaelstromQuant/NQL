# app.py

import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Noetherra Quant Lab", layout="wide")
st.title("🧪 Noetherra Strategies Quant Lab")
st.markdown("Backtest & visualize volatility-based strategies on hourly oil data.")

# Sidebar config
st.sidebar.header("Configuration")

symbol = st.sidebar.selectbox("Symbol", ['CL=F', 'BZ=F', 'NG=F'], index=0)
period = st.sidebar.selectbox("Period", ['30d', '90d', '180d'], index=1)
hold_hours = st.sidebar.slider("Holding Period (Hours)", 1, 12, 3)
vol_threshold = st.sidebar.slider("Volatility Threshold", 0.1, 1.0, 0.3, 0.05)
target_hour = st.sidebar.slider("Target Entry Hour (UTC)", 0, 23, 13)

# Fetch data
@st.cache_data
def fetch_data(symbol, period):
    return yf.download(tickers=symbol, period=period, interval="1h")

data = fetch_data(symbol, period)

if data.empty:
    st.error("Failed to fetch data.")
else:
    st.success(f"Loaded {symbol} data ({period})")
    st.line_chart(data['Close'])

    # Placeholder
    st.subheader("📈 Strategy Output")
    st.info("Strategy backtesting, metrics, and charts coming in next step!")
