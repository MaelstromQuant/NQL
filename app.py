import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# === CONFIGURATION ===
SYMBOL = 'CL=F'  # WTI Crude Oil
PERIOD = '90d'
INTERVAL = '1h'
HOLD_HOURS = 3
VOL_THRESHOLD = 0.3
ENTRY_HOUR = 13

# === FETCH DATA ===
@st.cache_data
def fetch_data(symbol, period, interval):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['Close'].rolling(window=3).std()
    df['hour'] = df.index.hour
    return df.dropna()

# === STRATEGY ===
def apply_strategy(df, entry_hour, vol_threshold, hold_hours):
    df['signal'] = ((df['hour'] == entry_hour) & (df['volatility'] > vol_threshold)).astype(int)
    df['position'] = 0
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1:
            end = min(i + hold_hours, len(df) - 1)
            df.iloc[i:end + 1, df.columns.get_loc('position')] = 1
    return df

# === BACKTEST ===
def backtest(df):
    df['strategy_return'] = df['position'].shift(1) * df['return']
    df['strategy_return'] -= 0.0002 * df['position'].diff().abs().fillna(0)
    df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
    df['buy_hold'] = (1 + df['return'].fillna(0)).cumprod()
    return df

# === MAIN APP ===
st.set_page_config(page_title="NQL Strategy Lab", layout="wide")
st.title("🧪 NQL Strategy Lab")
st.markdown("Backtest & visualize a volatility-driven intraday model.")

# Sidebar
symbol = st.sidebar.selectbox("Symbol", ['CL=F', 'BZ=F'], index=0)
period = st.sidebar.selectbox("Period", ['30d', '90d', '180d'], index=1)
hold_hours = st.sidebar.slider("Holding Hours", 1, 12, 3)
vol_thresh = st.sidebar.slider("Volatility Threshold", 0.1, 1.0, 0.3, 0.05)
entry_hour = st.sidebar.slider("Target Hour (UTC)", 0, 23, 13)

# Load and run
df = fetch_data(symbol, period, "1h")
df = apply_strategy(df, entry_hour, vol_thresh, hold_hours)
df = backtest(df)

# Display chart
st.subheader("📈 Strategy Performance vs Buy & Hold")
try:
    st.line_chart(df[['cumulative_return', 'buy_hold']])
except:
    st.warning("Chart failed to render. Check data integrity.")

# Show latest trades
st.subheader("🧾 Sample Strategy Data")
st.dataframe(df.tail(15))
