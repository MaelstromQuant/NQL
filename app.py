import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy.stats import linregress

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
def set_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #e9ecef;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        /* Metrics cards */
        .st-bh, .st-cg, .st-ci {
            background-color: white;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.1);
        }
        
        /* Titles */
        h1, h2, h3 {
            color: #1e3a8a;
        }
        
        /* Dataframes */
        .dataframe {
            border-radius: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.1);
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #4a6bdf;
            color: white;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

set_css()

# === CONFIGURATION ===
class Config:
    DEFAULT_HOLD_HOURS = 3
    DEFAULT_TRANSACTION_COST = 0.0002  # 2 basis points
    DEFAULT_VOL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
    DEFAULT_HOURS_TO_TEST = list(range(0, 24))
    DEFAULT_SYMBOL = 'CL=F'  # WTI Crude Oil
    DEFAULT_PERIOD = '180d'
    DEFAULT_INTERVAL = '1h'
    RISK_FREE_RATE = 0.0  # For Sharpe ratio calculation
    
    # Exchange session hour ranges in UTC
    SESSION_RANGES = {
        'Asia': range(0, 6),      # Tokyo/Singapore/HongKong
        'Europe': range(7, 16),   # London/Frankfurt overlap
        'US': range(13, 21),      # New York
        'Electronic': list(range(21, 24)) + list(range(0, 1))  # Overnight electronic
    }

# === DATA HANDLING ===
class DataHandler:
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Fetching market data...")
    def fetch_data(symbol, period, interval):
        """Fetch data with error handling and caching"""
        try:
            data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
            if data.empty:
                st.error("No data returned from Yahoo Finance")
                return None
            return data.dropna()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    @staticmethod
    def add_features(df):
        """Enhanced feature engineering"""
        if df is None or df.empty:
            return None
            
        df = df.copy()
        
        # Basic features
        df['return'] = df['Close'].pct_change()
        df['hour'] = df.index.hour
        df['date'] = df.index.date
        
        # Volatility measures
        df['volatility_3h'] = df['Close'].rolling(window=3).std()
        df['volatility_12h'] = df['Close'].rolling(window=12).std()
        df['volatility_ratio'] = df['volatility_3h'] / df['volatility_12h'].replace(0, np.nan)
        
        # Momentum indicators
        df['momentum_3h'] = df['Close'].pct_change(3)
        df['momentum_12h'] = df['Close'].pct_change(12)
        
        # Session features
        df = DataHandler.label_sessions(df)
        
        return df.dropna()

    @staticmethod
    def label_sessions(df):
        """Label trading sessions with overlap handling"""
        if df is None or df.empty:
            return None
            
        df = df.copy()
        df['session'] = 'Other'
        
        for name, hours in Config.SESSION_RANGES.items():
            mask = df['hour'].isin(hours)
            df.loc[mask, 'session'] = name
        
        # Handle overlapping sessions
        overlap_mask = (df['hour'] >= 13) & (df['hour'] < 16)  # London/NY overlap
        df.loc[overlap_mask, 'session'] = 'London/NY Overlap'
        
        return df

# === STRATEGY LOGIC ===
class TradingStrategy:
    @staticmethod
    def generate_signals(df, target_hour, vol_thresh, hold_hours):
        """Enhanced signal generation with multiple conditions"""
        if df is None or df.empty:
            return None
            
        df = df.copy()
        
        # Initialize signals
        df['signal'] = 0
        
        # Entry conditions
        entry_condition = (
            (df['hour'] == target_hour) & 
            (df['volatility_3h'] > vol_thresh) & 
            (df['volatility_ratio'] > 0.7)  # Recent volatility is significant
        )
        
        df.loc[entry_condition, 'signal'] = 1
        
        # Position management
        df['position'] = 0
        active_positions = 0
        
        for i in range(1, len(df)):
            if df['signal'].iloc[i] == 1 and active_positions == 0:
                # Enter position
                end_idx = min(i + hold_hours, len(df) - 1)
                df.iloc[i:end_idx+1, df.columns.get_loc('position')] = 1
                active_positions = 1
            elif active_positions > 0 and df['position'].iloc[i-1] == 1 and df['position'].iloc[i] == 0:
                # Position naturally expired
                active_positions = 0
        
        return df

# === BACKTESTING ===
class Backtester:
    @staticmethod
    def run_backtest(df, transaction_cost):
        """Comprehensive backtesting with enhanced metrics"""
        if df is None or df.empty:
            return None, pd.DataFrame(), {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'trades': 0
            }
            
        df = df.copy()
        
        # Calculate strategy returns
        df['strategy_return'] = df['position'].shift(1) * df['return']
        df['strategy_return'] -= transaction_cost * df['position'].diff().abs().fillna(0)
        
        # Cumulative returns
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
        df['buy_hold'] = (1 + df['return']).cumprod()
        
        # Trade logging
        trades = Backtester._log_trades(df)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Performance metrics
        metrics = Backtester._calculate_metrics(df, trades_df, transaction_cost)
        
        return df, trades_df, metrics

    @staticmethod
    def _log_trades(df):
        """Detailed trade logging with robust return calculation"""
        if df is None or df.empty:
            return []
            
        trades = []
        open_trade = None
        
        for i in range(1, len(df)):
            # Entry
            if df['position'].iloc[i-1] == 0 and df['position'].iloc[i] == 1:
                open_trade = {
                    'entry_time': df.index[i],
                    'entry_price': df['Close'].iloc[i],
                    'entry_session': df['session'].iloc[i],
                    'entry_volatility': df['volatility_3h'].iloc[i]
                }
            # Exit
            elif df['position'].iloc[i-1] == 1 and df['position'].iloc[i] == 0 and open_trade:
                try:
                    entry_price = float(open_trade['entry_price'])
                    exit_price = float(df['Close'].iloc[i])
                    
                    if entry_price == 0:
                        continue  # Skip invalid trades
                        
                    raw_return = (exit_price - entry_price) / entry_price
                    net_return = raw_return - 2*Config.DEFAULT_TRANSACTION_COST
                    
                    open_trade.update({
                        'exit_time': df.index[i],
                        'exit_price': exit_price,
                        'exit_session': df['session'].iloc[i],
                        'holding_period': (df.index[i] - open_trade['entry_time']).total_seconds() / 3600,
                        'return': raw_return,
                        'return_net': net_return if not np.isnan(net_return) else 0.0
                    })
                    trades.append(open_trade)
                except (TypeError, ValueError, KeyError):
                    continue
                finally:
                    open_trade = None
        
        return trades

    @staticmethod
    def _calculate_metrics(df, trades_df, transaction_cost):
        """Enhanced performance metrics with robust error handling"""
        if df is None or df.empty or len(trades_df) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'trades': 0
            }
            
        try:
            # Basic metrics
            total_return = float(df['cumulative_return'].iloc[-1]) - 1
            annualized_return = (1 + total_return) ** (365/(len(df)/24)) - 1
            
            # Risk-adjusted metrics
            excess_returns = df['strategy_return'] - Config.RISK_FREE_RATE/24
            excess_returns = excess_returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(excess_returns) == 0:
                sharpe = 0
            else:
                sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(24) if excess_returns.std() != 0 else 0
            
            downside_returns = excess_returns[excess_returns < 0]
            sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(24) if len(downside_returns) > 0 else 0
            
            # Drawdown
            cum_returns = df['cumulative_return']
            max_dd = (cum_returns / cum_returns.cummax() - 1).min()
            
            # Trade metrics
            valid_returns = trades_df['return_net'].replace([np.inf, -np.inf], np.nan).dropna()
            winning_trades = valid_returns[valid_returns > 0]
            losing_trades = valid_returns[valid_returns < 0]
            
            win_rate = len(winning_trades) / len(valid_returns) if len(valid_returns) > 0 else 0
            profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 else np.inf
            avg_trade_return = valid_returns.mean() if len(valid_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade_return': avg_trade_return,
                'trades': len(trades_df)
            }
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'trades': 0
            }

# === OPTIMIZATION ===
class Optimizer:
    @staticmethod
    def grid_search(raw_data, params):
        """Perform parameter optimization with walk-forward validation"""
        if raw_data is None or raw_data.empty:
            return pd.DataFrame()
            
        try:
            results = []
            
            # Split data into training and validation sets
            split_idx = int(len(raw_data) * 0.7)
            train_data = raw_data.iloc[:split_idx]
            valid_data = raw_data.iloc[split_idx:]
            
            # Parameter grid
            param_grid = {
                'hour': params['hours_to_test'],
                'vol_thresh': params['vol_thresholds'],
                'hold_hours': params['hold_hours']
            }
            
            # Train phase
            train_results = []
            for hour in param_grid['hour']:
                for vol in param_grid['vol_thresh']:
                    for hold in param_grid['hold_hours']:
                        df = DataHandler.add_features(train_data.copy())
                        if df is None:
                            continue
                            
                        df = TradingStrategy.generate_signals(df, hour, vol, hold)
                        if df is None:
                            continue
                            
                        _, _, metrics = Backtester.run_backtest(df, params['transaction_cost'])
                        
                        train_results.append({
                            'hour': hour,
                            'vol_thresh': vol,
                            'hold_hours': hold,
                            'train_sharpe': metrics['sharpe_ratio'],
                            'train_return': metrics['total_return']
                        })
            
            # Get top N configurations from training
            train_df = pd.DataFrame(train_results)
            if train_df.empty:
                return pd.DataFrame()
                
            top_configs = train_df.sort_values('train_sharpe', ascending=False).head(10)
            
            # Validate top configurations
            for _, config in top_configs.iterrows():
                df = DataHandler.add_features(valid_data.copy())
                if df is None:
                    continue
                    
                df = TradingStrategy.generate_signals(df, config['hour'], config['vol_thresh'], config['hold_hours'])
                if df is None:
                    continue
                    
                _, _, metrics = Backtester.run_backtest(df, params['transaction_cost'])
                
                results.append({
                    'hour': config['hour'],
                    'vol_thresh': config['vol_thresh'],
                    'hold_hours': config['hold_hours'],
                    'train_sharpe': config['train_sharpe'],
                    'train_return': config['train_return'],
                    'valid_sharpe': metrics['sharpe_ratio'],
                    'valid_return': metrics['total_return'],
                    'combined_sharpe': (config['train_sharpe'] + metrics['sharpe_ratio'])/2
                })
            
            results_df = pd.DataFrame(results)
            return results_df.sort_values('combined_sharpe', ascending=False)
            
        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            return pd.DataFrame()

# === VISUALIZATION ===
class Visualizer:
    @staticmethod
    def plot_equity_curve(df, config):
        """Plot equity curve with annotations"""
        if df is None or df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curves
        ax.plot(df.index, df['cumulative_return'], 
                label=f"Strategy (hour={config['hour']}, vol>{config['vol_thresh']:.2f}, hold={config['hold_hours']}h)")
        ax.plot(df.index, df['buy_hold'], label='Buy & Hold', alpha=0.7)
        
        # Add annotations
        ax.set_title('Strategy Performance', fontsize=14)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        
        # Formatting
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        return fig

    @staticmethod
    def plot_drawdown(df):
        """Plot drawdown curve"""
        if df is None or df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Calculate drawdown
        dd = (df['cumulative_return'] / df['cumulative_return'].cummax() - 1)
        
        # Plot
        ax.fill_between(df.index, dd, 0, color='red', alpha=0.3)
        ax.set_title('Drawdown', fontsize=14)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Formatting
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        return fig

    @staticmethod
    def plot_trade_analysis(trades_df):
        """Plot trade-level analysis"""
        if trades_df is None or trades_df.empty:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Trade returns distribution
        sns.histplot(trades_df['return_net'].replace([np.inf, -np.inf], np.nan).dropna(), 
                    bins=30, kde=True, ax=ax1)
        ax1.set_title('Distribution of Trade Returns', fontsize=14)
        ax1.set_xlabel('Return', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Returns by hour
        trades_df['entry_hour'] = trades_df['entry_time'].dt.hour
        hourly_returns = trades_df.groupby('entry_hour')['return_net'].mean()
        hourly_returns.plot(kind='bar', color='blue', alpha=0.6, ax=ax2)
        ax2.set_title('Average Returns by Entry Hour', fontsize=14)
        ax2.set_xlabel('Hour (UTC)', fontsize=12)
        ax2.set_ylabel('Average Return', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_session_returns(trades_df):
        """Plot returns by trading session"""
        if trades_df is None or trades_df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        session_returns = trades_df.groupby('entry_session')['return_net'] \
                                 .mean() \
                                 .sort_values()
        session_returns.plot(kind='barh', color='green', alpha=0.6, ax=ax)
        
        ax.set_title('Average Returns by Trading Session', fontsize=14)
        ax.set_xlabel('Average Return', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig

# === STREAMLIT UI ===
def main():
    st.title("📈 Volatility-Based Trading Strategy Dashboard")
    st.markdown("""
    This dashboard backtests a time-based trading strategy that enters positions at specific hours 
    when volatility exceeds a threshold. The strategy holds positions for a fixed number of hours.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Strategy Configuration")
    
    # Asset selection
    symbol = st.sidebar.text_input("Ticker Symbol", Config.DEFAULT_SYMBOL)
    period = st.sidebar.selectbox(
        "Data Period", 
        ['30d', '60d', '90d', '180d', '1y', '2y'], 
        index=3
    )
    interval = st.sidebar.selectbox(
        "Interval", 
        ['1h', '2h', '4h', '1d'], 
        index=0
    )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    default_hours = st.sidebar.multiselect(
        "Hours to Test (UTC)", 
        Config.DEFAULT_HOURS_TO_TEST, 
        default=[8, 13, 16]
    )
    vol_thresholds = st.sidebar.multiselect(
        "Volatility Thresholds", 
        Config.DEFAULT_VOL_THRESHOLDS, 
        default=[0.2, 0.3, 0.4]
    )
    hold_hours = st.sidebar.multiselect(
        "Holding Periods (hours)", 
        [1, 2, 3, 4, 6, 8, 12], 
        default=[2, 3, 4]
    )
    transaction_cost = st.sidebar.number_input(
        "Transaction Cost (%)", 
        min_value=0.0, 
        max_value=1.0, 
        value=Config.DEFAULT_TRANSACTION_COST*100, 
        step=0.01
    ) / 100
    
    # Manual backtest options
    st.sidebar.subheader("Manual Backtest")
    manual_hour = st.sidebar.selectbox("Entry Hour (UTC)", Config.DEFAULT_HOURS_TO_TEST, index=8)
    manual_vol = st.sidebar.selectbox("Volatility Threshold", Config.DEFAULT_VOL_THRESHOLDS, index=2)
    manual_hold = st.sidebar.selectbox("Holding Hours", [1, 2, 3, 4, 6, 8, 12], index=2)
    
    # Fetch data
    raw_data = DataHandler.fetch_data(symbol, period, interval)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Optimization", "Manual Backtest", "Data Exploration"])
    
    with tab1:
        st.header("Strategy Optimization")
        
        if st.button("Run Optimization"):
            if raw_data is None:
                st.error("No data available for optimization")
            else:
                params = {
                    'hours_to_test': default_hours,
                    'vol_thresholds': vol_thresholds,
                    'hold_hours': hold_hours,
                    'transaction_cost': transaction_cost
                }
                
                optimized_results = Optimizer.grid_search(raw_data, params)
                
                if optimized_results.empty:
                    st.warning("No valid configurations found. Try different parameters.")
                else:
                    # Display top 5 results
                    st.subheader("Top 5 Configurations")
                    st.dataframe(optimized_results.head().style.format({
                        'vol_thresh': '{:.3f}',
                        'train_sharpe': '{:.2f}',
                        'valid_sharpe': '{:.2f}',
                        'combined_sharpe': '{:.2f}',
                        'train_return': '{:.2%}',
                        'valid_return': '{:.2%}'
                    }))
                    
                    # Run backtest with best configuration
                    best_config = optimized_results.iloc[0].to_dict()
                    df = DataHandler.add_features(raw_data.copy())
                    df = TradingStrategy.generate_signals(
                        df, 
                        best_config['hour'], 
                        best_config['vol_thresh'], 
                        best_config['hold_hours']
                    )
                    df, trades_df, metrics = Backtester.run_backtest(df, transaction_cost)
                    
                    # Display metrics
                    st.subheader("Best Strategy Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{metrics['total_return']:.2%}")
                    col2.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
                    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    col2.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                    col3.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    col2.metric("Avg Trade Return", f"{metrics['avg_trade_return']:.2%}")
                    col3.metric("Total Trades", metrics['trades'])
                    
                    # Visualizations
                    st.subheader("Performance Charts")
                    equity_fig = Visualizer.plot_equity_curve(df, best_config)
                    if equity_fig:
                        st.pyplot(equity_fig)
                    
                    dd_fig = Visualizer.plot_drawdown(df)
                    if dd_fig:
                        st.pyplot(dd_fig)
                    
                    if not trades_df.empty:
                        st.subheader("Trade Analysis")
                        trade_fig = Visualizer.plot_trade_analysis(trades_df)
                        if trade_fig:
                            st.pyplot(trade_fig)
                        
                        session_fig = Visualizer.plot_session_returns(trades_df)
                        if session_fig:
                            st.pyplot(session_fig)
                        
                        st.subheader("Trade Log")
                        st.dataframe(trades_df.sort_values('entry_time', ascending=False))
    
    with tab2:
        st.header("Manual Backtest")
        
        if raw_data is None:
            st.error("No data available for backtesting")
        else:
            df = DataHandler.add_features(raw_data.copy())
            df = TradingStrategy.generate_signals(df, manual_hour, manual_vol, manual_hold)
            df, trades_df, metrics = Backtester.run_backtest(df, transaction_cost)
            
            # Display metrics
            st.subheader("Strategy Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{metrics['total_return']:.2%}")
            col2.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
            col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            col2.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            col3.metric("Win Rate", f"{metrics['win_rate']:.2%}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            col2.metric("Avg Trade Return", f"{metrics['avg_trade_return']:.2%}")
            col3.metric("Total Trades", metrics['trades'])
            
            # Visualizations
            st.subheader("Performance Charts")
            equity_fig = Visualizer.plot_equity_curve(df, {
                'hour': manual_hour,
                'vol_thresh': manual_vol,
                'hold_hours': manual_hold
            })
            if equity_fig:
                st.pyplot(equity_fig)
            
            dd_fig = Visualizer.plot_drawdown(df)
            if dd_fig:
                st.pyplot(dd_fig)
            
            if not trades_df.empty:
                st.subheader("Trade Analysis")
                trade_fig = Visualizer.plot_trade_analysis(trades_df)
                if trade_fig:
                    st.pyplot(trade_fig)
                
                session_fig = Visualizer.plot_session_returns(trades_df)
                if session_fig:
                    st.pyplot(session_fig)
                
                st.subheader("Trade Log")
                st.dataframe(trades_df.sort_values('entry_time', ascending=False))
    
    with tab3:
        st.header("Data Exploration")
        
        if raw_data is None:
            st.error("No data available for exploration")
        else:
            df = DataHandler.add_features(raw_data.copy())
            
            st.subheader("Raw Data")
            st.dataframe(df.tail())
            
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe())
            
            st.subheader("Hourly Returns Analysis")
            hourly_stats = df.groupby('hour')['return'].agg(['mean', 'std', 'count'])
            st.dataframe(hourly_stats.style.format({
                'mean': '{:.2%}',
                'std': '{:.2%}'
            }))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='hour', y='return', data=df, ax=ax)
            ax.set_title('Return Distribution by Hour (UTC)')
            ax.set_ylabel('Return')
            ax.set_xlabel('Hour (UTC)')
            st.pyplot(fig)
            
            st.subheader("Volatility Analysis")
            fig, ax = plt.subplots(figsize=(12, 6))
            df['volatility_3h'].plot(ax=ax)
            ax.set_title('3-Hour Rolling Volatility')
            ax.set_ylabel('Volatility')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
