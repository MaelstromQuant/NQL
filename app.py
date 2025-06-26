import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Noetherra Quant Lab", layout="wide")
st.title("🧪 Noetherra Strategies Quant Lab")
st.markdown("Backtest & visualize volatility-based strategies on hourly oil data.")

# === Sidebar Settings ===
st.sidebar.header("Configuration")

symbol = st.sidebar.selectbox("Symbol", ['CL=F', 'BZ=F'], index=0)
period = st.sidebar.selectbox("Period", ['30d', '90d', '180d'], index=1)
hold_hours = st.sidebar.slider("Holding Period (Hours)", 1, 12, 3)
vol_threshold = st.sidebar.slider("Volatility Threshold", 0.1, 1.0, 0.3, 0.05)
target_hour = st.sidebar.slider("Target Entry Hour (UTC)", 0, 23, 13)

@st.cache_data
def fetch_data(symbol, period):
    return yf.download(tickers=symbol, period=period, interval="1h")

df = fetch_data(symbol, period)

# === Feature Engineering ===
def add_features(df):
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['Close'].rolling(3).std()
    df['hour'] = df.index.hour
    df.dropna(inplace=True)
    return df

# === Strategy Logic ===
def generate_strategy(df, target_hour, vol_thresh, hold_hours):
    df['signal'] = (df['hour'] == target_hour) & (df['volatility'] > vol_thresh)
    df['position'] = 0

    for i in range(len(df)):
        if df['signal'].iloc[i]:
            end = min(i + hold_hours, len(df) - 1)
            df.loc[df.index[i:end], 'position'] = 1

    df['strategy_return'] = df['position'].shift(1) * df['return']
    df['strategy_return'] -= 0.0002 * df['position'].diff().abs().fillna(0)
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['buy_hold'] = (1 + df['return']).cumprod()
    return df

if df.empty:
    st.error("Failed to fetch data.")
else:
    st.success(f"Loaded {symbol} data ({period})")

    df = add_features(df)
    df = generate_strategy(df, target_hour, vol_threshold, hold_hours)

    # === Plot Strategy ===
    st.subheader("📈 Strategy Backtest vs Buy & Hold")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['cumulative_return'], label='Strategy')
    ax.plot(df.index, df['buy_hold'], label='Buy & Hold', linestyle='--')
    ax.set_title("Cumulative Return Comparison")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # === Metrics ===
    st.subheader("📊 Performance Metrics")
    total_return = df['cumulative_return'].iloc[-1] - 1
    bh_return = df['buy_hold'].iloc[-1] - 1
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(24)

    st.metric("Strategy Return", f"{total_return:.2%}")
    st.metric("Buy & Hold Return", f"{bh_return:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
