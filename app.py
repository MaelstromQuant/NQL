import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# === CONFIG ===
TRANSACTION_COST = 0.0002

# === SETUP STREAMLIT ===
st.set_page_config(page_title="NQL | Quant Lab", layout="wide")
st.title("🧪 NQL | Quant Lab")
st.markdown("Backtest & visualize volatility-based strategies on hourly oil data.")

# === SIDEBAR CONFIG ===
st.sidebar.header("Configuration")
symbol = st.sidebar.selectbox("Symbol", ['CL=F', 'BZ=F', 'NG=F'], index=0)
period = st.sidebar.selectbox("Period", ['30d', '90d', '180d'], index=1)
hold_hours = st.sidebar.slider("Holding Period (Hours)", 1, 12, 3)
vol_threshold = st.sidebar.slider("Volatility Threshold", 0.1, 1.0, 0.3, 0.05)
target_hour = st.sidebar.slider("Target Entry Hour (UTC)", 0, 23, 13)

# === FETCH DATA ===
@st.cache_data
def fetch_data(symbol, period):
    df = yf.download(tickers=symbol, period=period, interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

data = fetch_data(symbol, period)

if data.empty or 'Close' not in data.columns:
    st.error("Failed to load valid price data.")
    st.stop()

# === DISPLAY RAW PRICE ===
st.success(f"✅ Loaded {symbol} data ({period})")
st.line_chart(data[['Close']])

# === FEATURE ENGINEERING ===
df = data.copy()
df['return'] = df['Close'].pct_change()
df['hour'] = df.index.hour
df['volatility'] = df['Close'].rolling(window=3).std()

# === SIGNALS & POSITIONS ===
df['signal'] = ((df['hour'] == target_hour) & (df['volatility'] > vol_threshold)).astype(int)
df['position'] = 0

for i in range(len(df)):
    if df['signal'].iloc[i] == 1:
        end_idx = min(i + hold_hours, len(df) - 1)
        df.iloc[i:end_idx + 1, df.columns.get_loc('position')] = 1

# === STRATEGY RETURNS ===
df['strategy_return'] = df['position'].shift(1) * df['return']
df['strategy_return'] -= TRANSACTION_COST * df['position'].diff().abs().fillna(0)
df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
df['buy_hold'] = (1 + df['return'].fillna(0)).cumprod()

# === PLOT RETURNS ===
st.subheader("📈 Strategy Performance")
plot_df = df[['cumulative_return', 'buy_hold']].dropna()
st.line_chart(plot_df)

# === TRADE LOG ===
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
        trades.append(open_trade)
        open_trade = None

trades_df = pd.DataFrame(trades)

st.subheader("🧾 Sample Trades")
if not trades_df.empty:
    st.dataframe(trades_df.head())
else:
    st.warning("No trades found for selected configuration.")
