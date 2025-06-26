# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# === PAGE SETUP ===
st.set_page_config(page_title="NQL – Noetherra Quant Lab", layout="wide")
st.title("📈 NQL – Noetherra Quant Lab")
st.markdown("Test volatility-driven models across time zones using hourly market data.")

# === SIDEBAR CONFIG ===
st.sidebar.header("⚙️ Strategy Configuration")
symbol = st.sidebar.selectbox("Market Symbol", ['CL=F', 'BZ=F', 'NG=F'], index=0)
period = st.sidebar.selectbox("Historical Period", ['30d', '90d', '180d'], index=1)
entry_hour = st.sidebar.slider("Target Entry Hour (UTC)", 0, 23, 13)
hold_hours = st.sidebar.slider("Holding Period (Hours)", 1, 12, 3)
vol_threshold = st.sidebar.slider("Volatility Threshold", 0.05, 1.0, 0.3, 0.05)

# === FETCH DATA ===
@st.cache_data
def fetch_data(symbol, period):
    df = yf.download(tickers=symbol, period=period, interval="1h", progress=False)
    df = df.dropna()
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['Close'].rolling(window=3).std()
    df['hour'] = df.index.hour
    return df.dropna()

df = fetch_data(symbol, period)

if df.empty:
    st.error("No data found. Please try a different symbol or period.")
    st.stop()

# === STRATEGY LOGIC ===
def apply_strategy(df, entry_hour, vol_threshold, hold_hours):
    df = df.copy()
    df['signal'] = ((df['hour'] == entry_hour) & (df['volatility'] > vol_threshold)).astype(int)
    df['position'] = 0
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1:
            end = min(i + hold_hours, len(df) - 1)
            df.iloc[i:end+1, df.columns.get_loc('position')] = 1
    return df

df = apply_strategy(df, entry_hour, vol_threshold, hold_hours)

# === BACKTEST ===
def run_backtest(df):
    df = df.copy()
    df['strategy_return'] = df['position'].shift(1) * df['return']
    df['strategy_return'] -= 0.0002 * df['position'].diff().abs().fillna(0)
    df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
    df['buy_hold'] = (1 + df['return'].fillna(0)).cumprod()
    return df

df = run_backtest(df)

# === TRADE LOG ===
def get_trade_log(df):
    trades = []
    open_trade = None
    for i in range(1, len(df)):
        if df['position'].iloc[i-1] == 0 and df['position'].iloc[i] == 1:
            open_trade = {'entry_time': df.index[i], 'entry_price': df['Close'].iloc[i]}
        elif df['position'].iloc[i-1] == 1 and df['position'].iloc[i] == 0 and open_trade:
            open_trade['exit_time'] = df.index[i]
            open_trade['exit_price'] = df['Close'].iloc[i]
            open_trade['return'] = (open_trade['exit_price'] - open_trade['entry_price']) / open_trade['entry_price']
            trades.append(open_trade)
            open_trade = None
    trades_df = pd.DataFrame(trades)
    trades_df['return'] = pd.to_numeric(trades_df['return'], errors='coerce')
    trades_df.dropna(subset=['return'], inplace=True)
    return trades_df

trades_df = get_trade_log(df)

# === METRICS ===
if not trades_df.empty:
    total_return = df['cumulative_return'].iloc[-1] - 1
    annualized_return = (1 + total_return)**(365/(len(df)/24)) - 1
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(24) if df['strategy_return'].std() > 0 else 0
    max_dd = (df['cumulative_return'] / df['cumulative_return'].cummax() - 1).min()
    win_rate = (trades_df['return'] > 0).mean()
    profit_factor = trades_df[trades_df['return'] > 0]['return'].sum() / abs(trades_df[trades_df['return'] < 0]['return'].sum()) if not trades_df[trades_df['return'] < 0].empty else np.inf
else:
    total_return = annualized_return = sharpe = max_dd = win_rate = profit_factor = 0

# === RESULTS ===
col1, col2, col3 = st.columns(3)
col1.metric("📈 Total Return", f"{total_return:.2%}")
col1.metric("⏱ Annualized Return", f"{annualized_return:.2%}")
col2.metric("📊 Sharpe Ratio", f"{sharpe:.2f}")
col2.metric("✅ Win Rate", f"{win_rate:.2%}")
col3.metric("📉 Max Drawdown", f"{max_dd:.2%}")
col3.metric("💰 Profit Factor", f"{profit_factor:.2f}")

# === CHARTS ===
st.subheader("📉 Strategy vs. Buy & Hold")
st.line_chart(df[['cumulative_return', 'buy_hold']])

# === TRADE LOG ===
st.subheader("📋 Sample Trades")
st.dataframe(trades_df.head(10))
