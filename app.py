import streamlit as st
from datetime import datetime
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Streamlit setup
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
transaction_cost = 0.0002  # 2 basis points

# Data fetching
@st.cache_data
def fetch_data(symbol, period):
    return yf.download(tickers=symbol, period=period, interval="1h")

data = fetch_data(symbol, period)

if data.empty:
    st.error("Failed to fetch data.")
else:
    st.success(f"Loaded {symbol} data ({period})")
    st.line_chart(data['Close'])

    # Feature engineering
    df = data.copy()
    df['return'] = df['Close'].pct_change()
    df['hour'] = df.index.hour
    df['volatility'] = df['Close'].rolling(window=3).std()

    # Signal generation
    df['signal'] = ((df['hour'] == target_hour) & (df['volatility'] > vol_threshold)).astype(int)
    df['position'] = 0

    for i in range(len(df)):
        if df['signal'].iloc[i] == 1:
            end_idx = min(i + hold_hours, len(df) - 1)
            df.iloc[i:end_idx + 1, df.columns.get_loc('position')] = 1

    # Strategy returns
    df['strategy_return'] = df['position'].shift(1) * df['return']
    df['strategy_return'] -= transaction_cost * df['position'].diff().abs().fillna(0)
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['buy_hold'] = (1 + df['return']).cumprod()

    # Trade logging
    trades = []
    open_trade = None
    for i in range(1, len(df)):
        if df['position'].iloc[i - 1] == 0 and df['position'].iloc[i] == 1:
            open_trade = {
                'entry_time': df.index[i],
                'entry_price': df['Close'].iloc[i]
            }
        elif df['position'].iloc[i - 1] == 1 and df['position'].iloc[i] == 0 and open_trade:
            open_trade['exit_time'] = df.index[i]
            open_trade['exit_price'] = df['Close'].iloc[i]
            open_trade['return'] = (open_trade['exit_price'] - open_trade['entry_price']) / open_trade['entry_price']
            open_trade['holding_hours'] = (open_trade['exit_time'] - open_trade['entry_time']).seconds / 3600
            trades.append(open_trade)
            open_trade = None

    trades_df = pd.DataFrame(trades)

    # Plotting
    st.subheader("📈 Strategy Performance")
    st.line_chart(df[['cumulative_return', 'buy_hold']])

    # Display sample trades
    st.subheader("📄 Sample Trades")
    if not trades_df.empty:
        st.dataframe(trades_df.head())
    else:
        st.warning("No trades generated in this configuration.")
