# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="NQL – Quant Lab", layout="wide")
st.title("📈 NQL – Quant Strategy Lab")
st.markdown("Backtest hourly volatility-based strategies on oil futures (WTI, Brent, etc.).")

# === SIDEBAR CONFIG ===
st.sidebar.header("🔧 Strategy Config")

symbol = st.sidebar.selectbox("Symbol", ['CL=F', 'BZ=F', 'NG=F'], index=0)
period = st.sidebar.selectbox("Data Period", ['30d', '90d', '180d'], index=2)
hold_hours = st.sidebar.slider("Holding Period (Hours)", 1, 12, 3)
vol_thresh = st.sidebar.slider("Volatility Threshold", 0.1, 1.0, 0.3, step=0.05)
entry_hour = st.sidebar.slider("Entry Hour (UTC)", 0, 23, 13)

# === FETCH DATA ===
@st.cache_data
def load_data(symbol, period):
    df = yf.download(tickers=symbol, period=period, interval='1h')
    df.dropna(inplace=True)
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['Close'].rolling(window=3).std()
    df['hour'] = df.index.hour
    df['session'] = 'Other'
    df.loc[df['hour'].between(0, 6), 'session'] = 'Asia'
    df.loc[df['hour'].between(7, 12), 'session'] = 'Europe'
    df.loc[df['hour'].between(13, 20), 'session'] = 'US'
    df.loc[df['hour'].between(13, 15), 'session'] = 'Overlap'
    return df.dropna()

data = load_data(symbol, period)

if data.empty:
    st.error("⚠️ No data loaded.")
    st.stop()

# === STRATEGY LOGIC ===
def generate_strategy(df, hour, vol_thresh, hold_hours):
    df = df.copy()
    df['signal'] = ((df['hour'] == hour) & (df['volatility'] > vol_thresh)).astype(int)
    df['position'] = 0
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1:
            end_idx = min(i + hold_hours, len(df) - 1)
            df.iloc[i:end_idx + 1, df.columns.get_loc('position')] = 1
    return df

def run_backtest(df):
    df['strategy_return'] = df['position'].shift(1) * df['return']
    df['strategy_return'] -= 0.0002 * df['position'].diff().abs().fillna(0)
    df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
    df['buy_hold'] = (1 + df['return'].fillna(0)).cumprod()
    return df

def get_trade_log(df):
    trades = []
    open_trade = None
    for i in range(1, len(df)):
        if df['position'].iloc[i - 1] == 0 and df['position'].iloc[i] == 1:
            open_trade = {
                'entry_time': df.index[i],
                'entry_price': df['Close'].iloc[i],
                'session': df['session'].iloc[i]
            }
        elif df['position'].iloc[i - 1] == 1 and df['position'].iloc[i] == 0 and open_trade:
            open_trade['exit_time'] = df.index[i]
            open_trade['exit_price'] = df['Close'].iloc[i]
            open_trade['return'] = (open_trade['exit_price'] - open_trade['entry_price']) / open_trade['entry_price']
            trades.append(open_trade)
            open_trade = None
    trades_df = pd.DataFrame(trades)
    # Ensure numeric return
    trades_df['return'] = pd.to_numeric(trades_df['return'], errors='coerce')
    trades_df.dropna(subset=['return'], inplace=True)
    return trades_df

# === STRATEGY EXECUTION ===
df = generate_strategy(data.copy(), entry_hour, vol_thresh, hold_hours)
df = run_backtest(df)
trades_df = get_trade_log(df)

# === PERFORMANCE METRICS ===
st.subheader("📊 Performance Metrics")

if not trades_df.empty:
    total_return = df['cumulative_return'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (365 / (len(df) / 24)) - 1
    strategy_returns = df['strategy_return'].dropna()
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(24) if strategy_returns.std() > 0 else 0
    max_drawdown = ((df['cumulative_return'] / df['cumulative_return'].cummax()) - 1).min()
    win_rate = (trades_df['return'] > 0).mean()
    profit_factor = trades_df[trades_df['return'] > 0]['return'].sum() / abs(trades_df[trades_df['return'] < 0]['return'].sum()) if not trades_df[trades_df['return'] < 0].empty else np.inf

    st.metric("📈 Total Return", f"{total_return:.2%}")
    st.metric("⏱ Annualized Return", f"{annualized_return:.2%}")
    st.metric("📊 Sharpe Ratio", f"{sharpe_ratio:.2f}")
    st.metric("✅ Win Rate", f"{win_rate:.2%}")
    st.metric("📉 Max Drawdown", f"{max_drawdown:.2%}")
    st.metric("💰 Profit Factor", f"{profit_factor:.2f}")
else:
    st.warning("⚠️ No trades found under these settings.")

# === CHARTS ===
st.subheader("📉 Strategy vs Buy & Hold")
st.line_chart(df[['cumulative_return', 'buy_hold']])

# === TRADE LOG ===
st.subheader("📄 Sample Trade Log")
st.dataframe(trades_df.head())

# === OPTIMIZATION (Optional)
if st.checkbox("🔍 Run Optimization (Grid Search)", value=False):
    with st.spinner("Running optimization..."):
        results = []
        for h in range(24):
            for v in np.linspace(0.1, 0.5, 5):
                temp_df = generate_strategy(data.copy(), h, v, hold_hours)
                temp_df = run_backtest(temp_df)
                final_ret = temp_df['cumulative_return'].iloc[-1]
                sharpe = temp_df['strategy_return'].mean() / temp_df['strategy_return'].std() * np.sqrt(24) if temp_df['strategy_return'].std() > 0 else 0
                max_dd = (temp_df['cumulative_return'] / temp_df['cumulative_return'].cummax() - 1).min()
                trades = get_trade_log(temp_df)
                results.append({
                    'hour': h,
                    'vol_thresh': round(v, 2),
                    'return': final_ret,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd,
                    'trades': len(trades)
                })
        opt_df = pd.DataFrame(results)
        st.success("Optimization complete.")
        st.dataframe(opt_df.sort_values(by='sharpe', ascending=False).reset_index(drop=True))
