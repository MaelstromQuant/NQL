import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from gym import Env, spaces
import time
import pytz
from collections import deque
import plotly.graph_objects as go
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import threading

# Configuration
class Config:
    ASSETS = {
        'WTI Crude Oil': 'CL=F',
        'Natural Gas': 'NG=F',
        'Brent Crude': 'BZ=F'
    }
    DEFAULT_PERIOD = "6mo"
    DEFAULT_INTERVAL = "1h"
    VOL_THRESH_RANGE = (0.001, 0.01)  # For grid search
    HOLD_HOURS_RANGE = (1, 24)        # For grid search
    TRANSACTION_COST = 0.0005
    NEWS_UPDATE_MINUTES = 15
    EIA_UPDATE_HOURS = 24
    SENTIMENT_TRIGGERS = {
        'Strong Positive': 0.5,
        'Moderate Positive': 0.3,
        'Neutral': 0,
        'Moderate Negative': -0.3,
        'Strong Negative': -0.5
    }

# Enhanced Data Handler
class DataHandler:
    @staticmethod
    def fetch_data(ticker, period, interval):
        """Fetch price data from Yahoo Finance with error handling"""
        try:
            data = yf.download(tickers=ticker, period=period, interval=interval)
            if data.empty:
                st.error(f"No data returned for {ticker}. Please check parameters.")
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None

    @staticmethod
    def add_features(df, ticker):
        """Add technical and temporal features to the dataframe"""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Basic price transformations
        df['return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close']/df['Close'].shift(1))
        
        # Volatility features
        for window in [3, 6, 12, 24]:
            df[f'volatility_{window}h'] = df['return'].rolling(window).std()
            df[f'volatility_ratio_{window}h'] = (
                df[f'volatility_{window//3}h'] / df[f'volatility_{window}h'])
        
        # Momentum features
        for window in [3, 6, 12, 24]:
            df[f'momentum_{window}h'] = df['Close'].pct_change(window)
            df[f'ma_{window}h'] = df['Close'].rolling(window).mean()
            df[f'ema_{window}h'] = df['Close'].ewm(span=window).mean()
        
        # Session labels (Asia, Europe, US)
        df['hour'] = df.index.hour
        df['session'] = 'Other'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'Asia'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'Europe'
        df.loc[(df['hour'] >= 16) & (df['hour'] < 24), 'session'] = 'US'
        
        # Day of week effects
        df['day_of_week'] = df.index.dayofweek
        
        # Asset-specific features
        if ticker == 'NG=F':  # Natural Gas
            df['season'] = df.index.month % 12 // 3 + 1  # 1=Winter, 2=Spring, etc.
        
        return df

    @staticmethod
    def prepare_lstm_data(df, lookback=60):
        """Prepare data for LSTM forecasting"""
        if df is None or len(df) < lookback * 2:
            return None, None, None
            
        # Use log returns as more normally distributed
        data = df['log_return'].dropna().values.reshape(-1, 1)
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaler

# Enhanced News Handler with Live Triggers
class NewsHandler:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_update = None
        self.cached_sentiment = 0
        self.cached_news = []
        self.trigger_events = deque(maxlen=10)
        self.lock = threading.Lock()
        
    def fetch_latest_headlines(self, query="crude oil OR WTI OR natural gas OR brent", page_size=20):
        """Fetch news headlines with expanded query"""
        if not self.api_key:
            return []
            
        try:
            url = (f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}"
                   f"&sortBy=publishedAt&apiKey={self.api_key}")
            response = requests.get(url)
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return []
    
    def update_news_sentiment(self):
        """Update news sentiment and check for triggers"""
        if not self.api_key:
            return 0
            
        now = datetime.now()
        if (self.last_update is None or 
            (now - self.last_update).total_seconds() > Config.NEWS_UPDATE_MINUTES * 60):
            
            articles = self.fetch_latest_headlines()
            if articles:
                with self.lock:
                    self.cached_news = articles
                    scores = []
                    for article in articles:
                        if article.get('title'):
                            score = self.analyzer.polarity_scores(article['title'])
                            scores.append(score['compound'])
                            # Check for triggers
                            if score['compound'] >= Config.SENTIMENT_TRIGGERS['Strong Positive']:
                                self.trigger_events.append({
                                    'type': 'bullish_trigger',
                                    'timestamp': now,
                                    'title': article['title'],
                                    'score': score['compound']
                                })
                            elif score['compound'] <= Config.SENTIMENT_TRIGGERS['Strong Negative']:
                                self.trigger_events.append({
                                    'type': 'bearish_trigger',
                                    'timestamp': now,
                                    'title': article['title'],
                                    'score': score['compound']
                                })
                    
                    if scores:
                        self.cached_sentiment = np.mean(scores)
                    self.last_update = now
                
        return self.cached_sentiment

    def get_recent_triggers(self):
        """Get recent sentiment triggers"""
        with self.lock:
            return list(self.trigger_events)

# LSTM Forecaster
class LSTMForecaster:
    def __init__(self, lookback=60, units=50):
        self.lookback = lookback
        self.units = units
        self.model = None
        self.scaler = None
        
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            LSTM(self.units),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X, y, epochs=20, batch_size=32):
        """Train LSTM model"""
        if X is None or y is None:
            return False
            
        self.model = self.build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return True
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or X is None:
            return None
        return self.model.predict(X)

# Enhanced Trading Strategy
class TradingStrategy:
    @staticmethod
    def generate_signals(df, params, use_sentiment=False, use_lstm=False, lstm_forecaster=None):
        """Generate trading signals with multiple features"""
        if df is None or df.empty:
            return None
            
        df = df.copy()
        df['signal'] = 0
        
        # Base volatility condition
        vol_thresh = params.get('vol_thresh', 0.0025)
        hold_hours = params.get('hold_hours', 6)
        target_hour = params.get('target_hour', 10)
        
        base_condition = (
            (df['hour'] == target_hour) & 
            (df['volatility_3h'] > vol_thresh) & 
            (df['volatility_ratio_12h'] > params.get('vol_ratio_thresh', 0.7))
        )
        
        # Sentiment condition
        if use_sentiment and 'news_sentiment' in df:
            sentiment_condition = (
                df['news_sentiment'] > params.get('sentiment_thresh', -0.5)
            )
            entry_condition = base_condition & sentiment_condition
        else:
            entry_condition = base_condition
        
        # LSTM forecast condition
        if use_lstm and lstm_forecaster and 'lstm_forecast' in df:
            lstm_condition = df['lstm_forecast'] > params.get('lstm_thresh', 0)
            entry_condition = entry_condition & lstm_condition
        
        # Generate signals
        df.loc[entry_condition, 'signal'] = 1
        
        # Implement hold period
        for i in df[df['signal'] == 1].index:
            end_idx = min(len(df), df.index.get_loc(i) + hold_hours)
            df.loc[i:df.index[end_idx-1], 'position'] = 1
        
        # Fill remaining positions with 0
        df['position'] = df['position'].fillna(0)
        
        return df

    @staticmethod
    def generate_sentiment_trigger_signals(df, triggers):
        """Generate signals based on sentiment triggers"""
        if df is None or not triggers:
            return df
            
        df = df.copy()
        if 'signal' not in df:
            df['signal'] = 0
            
        for trigger in triggers:
            trigger_time = trigger['timestamp']
            if trigger['type'] == 'bullish_trigger':
                # Find the next market open after trigger
                next_idx = df.index[df.index >= trigger_time]
                if len(next_idx) > 0:
                    df.loc[next_idx[0], 'signal'] = 1
            elif trigger['type'] == 'bearish_trigger':
                # Find the next market open after trigger
                next_idx = df.index[df.index >= trigger_time]
                if len(next_idx) > 0:
                    df.loc[next_idx[0], 'signal'] = -1  # Short signal
        
        return df

# Backtesting Optimizer
class BacktestOptimizer:
    @staticmethod
    def grid_search(df, param_grid):
        """Perform grid search over parameter space"""
        if df is None:
            return None
            
        best_params = None
        best_sharpe = -np.inf
        results = []
        
        # Generate all parameter combinations
        all_params = list(ParameterGrid(param_grid))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, params in enumerate(all_params):
            # Update progress
            progress = (i + 1) / len(all_params)
            progress_bar.progress(progress)
            status_text.text(f"Testing combination {i+1}/{len(all_params)}: {params}")
            
            # Run backtest with current params
            signals = TradingStrategy.generate_signals(df, params)
            if signals is None:
                continue
                
            backtest_df = Backtester.run_backtest(signals, Config.TRANSACTION_COST)
            if backtest_df is None:
                continue
                
            metrics = Backtester.calculate_metrics(backtest_df)
            
            # Track results
            results.append({
                'params': params,
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['strategy_return'],
                'drawdown': metrics['max_drawdown']
            })
            
            # Update best params
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = params
        
        progress_bar.empty()
        status_text.empty()
        
        return best_params, pd.DataFrame(results)

# Enhanced Backtester
class Backtester:
    @staticmethod
    def run_backtest(df, transaction_cost=0.0005):
        """Run backtest on strategy with enhanced metrics"""
        if df is None or 'position' not in df:
            return None
            
        df = df.copy()
        
        # Calculate strategy returns
        df['strategy_return'] = df['position'].shift(1) * df['return']
        
        # Apply transaction costs
        trade_dates = df[df['position'].diff() != 0].index
        df.loc[trade_dates, 'strategy_return'] -= transaction_cost
        
        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
        df['benchmark_return'] = (1 + df['return']).cumprod()
        
        # Calculate rolling Sharpe ratio (6 month window)
        df['rolling_sharpe'] = (
            df['strategy_return'].rolling(30).mean() / 
            df['strategy_return'].rolling(30).std() * np.sqrt(252)
        )
        
        return df
    
    @staticmethod
    def calculate_metrics(df):
        """Calculate comprehensive performance metrics"""
        if df is None or 'strategy_return' not in df:
            return {}
            
        metrics = {}
        
        # Returns
        metrics['strategy_return'] = df['cumulative_return'].iloc[-1] - 1
        metrics['benchmark_return'] = df['benchmark_return'].iloc[-1] - 1
        
        # Annualized metrics
        days = (df.index[-1] - df.index[0]).days
        metrics['annualized_strategy'] = (1 + metrics['strategy_return'])**(365/days) - 1
        metrics['annualized_benchmark'] = (1 + metrics['benchmark_return'])**(365/days) - 1
        
        # Risk metrics
        metrics['strategy_volatility'] = df['strategy_return'].std() * np.sqrt(252)
        metrics['benchmark_volatility'] = df['return'].std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_strategy'] / metrics['strategy_volatility']
        metrics['sortino_ratio'] = Backtester.calculate_sortino(df['strategy_return'])
        
        # Drawdown
        cum_returns = df['cumulative_return']
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown.mean()
        metrics['drawdown_duration'] = Backtester.calculate_drawdown_duration(drawdown)
        
        # Trade statistics
        trades = df[df['position'].diff() != 0]
        metrics['num_trades'] = len(trades)
        
        if metrics['num_trades'] > 0:
            wins = df[df['strategy_return'] > 0]['strategy_return'].count()
            metrics['win_rate'] = wins / metrics['num_trades']
            metrics['avg_win'] = df[df['strategy_return'] > 0]['strategy_return'].mean()
            metrics['avg_loss'] = df[df['strategy_return'] < 0]['strategy_return'].mean()
            metrics['profit_factor'] = abs(
                metrics['avg_win'] * wins / 
                (metrics['avg_loss'] * (metrics['num_trades'] - wins))
        else:
            metrics['win_rate'] = 0
            metrics['avg_win'] = 0
            metrics['avg_loss'] = 0
            metrics['profit_factor'] = 0
        
        return metrics
    
    @staticmethod
    def calculate_sortino(returns, risk_free=0):
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < risk_free]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        if downside_volatility == 0:
            return np.nan
        excess_returns = returns.mean() * 252 - risk_free
        return excess_returns / downside_volatility
    
    @staticmethod
    def calculate_drawdown_duration(drawdown_series):
        """Calculate average drawdown duration"""
        drawdown_periods = (drawdown_series < 0).astype(int)
        duration = drawdown_periods * (drawdown_periods.groupby(
            (drawdown_periods != drawdown_periods.shift()).cumsum()).cumcount() + 1)
        return duration.mean()

# Enhanced RL Environment
class OilTradingEnv(Env):
    def __init__(self, df, initial_balance=100000, transaction_cost=0.0005):
        super().__init__()
        self.df = df
        self.current_step = 0
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Enhanced state space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(20,), dtype=np.float32  # Expanded state space
        )
        
        # Action space: 0=hold, 1=buy, 2=sell, 3=strong buy, 4=strong sell
        self.action_space = spaces.Discrete(5)
        
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 20  # Start with enough history
        self.trades = []
        self.portfolio_values = [self.balance]
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state vector with enhanced features"""
        # Past 10 returns
        returns = self.df['return'].iloc[self.current_step-10:self.current_step].values
        
        # Current features
        features = [
            self.df['volatility_3h'].iloc[self.current_step],
            self.df['volatility_ratio_12h'].iloc[self.current_step],
            self.df['momentum_6h'].iloc[self.current_step],
            self.df['momentum_12h'].iloc[self.current_step],
            self.df.get('news_sentiment', 0).iloc[self.current_step],
            self.df.get('inventory_level', 0).iloc[self.current_step],
            self.df.get('lstm_forecast', 0).iloc[self.current_step],
            self.position,
            self.balance / self.initial_balance,  # Normalized balance
            (self.df['Close'].iloc[self.current_step] - 
             self.df['Close'].iloc[self.current_step-1]) / 
             self.df['Close'].iloc[self.current_step-1]  # Latest price change
        ]
        
        state = np.concatenate([returns, features]).astype(np.float32)
        return state
    
    def step(self, action):
        """Execute one step in the environment with enhanced actions"""
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0, True, {}
            
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0
        info = {'action': 'hold'}
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.transaction_cost * self.balance
                info['action'] = 'buy'
        elif action == 2:  # Sell
            if self.position == 1:
                profit = (current_price - self.entry_price) / self.entry_price
                reward = profit * 100  # Scale reward
                self.balance *= (1 + profit - self.transaction_cost)
                self.position = 0
                info['action'] = 'sell'
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'profit': profit,
                    'step': self.current_step
                })
        elif action == 3:  # Strong buy (2x position)
            if self.position == 0:
                self.position = 2  # Double position
                self.entry_price = current_price
                self.balance -= self.transaction_cost * self.balance
                info['action'] = 'strong_buy'
        elif action == 4:  # Strong sell
            if self.position > 0:
                profit = (current_price - self.entry_price) / self.entry_price * self.position
                reward = profit * 100
                self.balance *= (1 + profit - self.transaction_cost)
                self.position = 0
                info['action'] = 'strong_sell'
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'profit': profit,
                    'step': self.current_step,
                    'size': self.position
                })
        
        # Move to next step
        self.current_step += 1
        self.portfolio_values.append(self.balance)
        
        # Additional reward for portfolio growth
        reward += (self.balance / self.initial_balance - 1) * 10
        
        # Penalize for drawdowns
        current_max = max(self.portfolio_values)
        drawdown = (current_max - self.balance) / current_max
        reward -= drawdown * 5
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Get new state
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done, info

# Streamlit App
def main():
    st.set_page_config(
        page_title="Multi-Asset Trading Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Multi-Asset Trading Dashboard")
    st.markdown("""
    Advanced trading dashboard with support for multiple commodities, LSTM forecasting, 
    sentiment triggers, and reinforcement learning.
    """)
    
    # Initialize session state
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    if 'backtest_data' not in st.session_state:
        st.session_state.backtest_data = {}
    if 'rl_model' not in st.session_state:
        st.session_state.rl_model = {}
    if 'rl_results' not in st.session_state:
        st.session_state.rl_results = {}
    if 'lstm_models' not in st.session_state:
        st.session_state.lstm_models = {}
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API keys
    st.sidebar.subheader("API Keys")
    news_api_key = st.sidebar.text_input("NewsAPI Key", type="password")
    eia_api_key = st.sidebar.text_input("EIA API Key", type="password")
    
    # Asset selection
    st.sidebar.subheader("Asset Selection")
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        list(Config.ASSETS.keys()),
        default=['WTI Crude Oil']
    )
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    interval = st.sidebar.selectbox(
        "Interval",
        ['5m', '15m', '30m', '1h', '2h', '4h', '1d'],
        index=3
    )
    
    # Adjust period based on interval
    if interval in ['5m', '15m']:
        period_options = ['1d', '5d', '1mo', '3mo']
    else:
        period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y']
    
    period = st.sidebar.selectbox("Period", period_options, index=3)
    
    # Initialize handlers
    news_handler = NewsHandler(news_api_key)
    econ_handler = EconDataHandler(eia_api_key)
    
    # Data fetching
    st.sidebar.subheader("Data Actions")
    if st.sidebar.button("Fetch Data for Selected Assets"):
        with st.spinner("Fetching data..."):
            for asset_name in selected_assets:
                ticker = Config.ASSETS[asset_name]
                raw_data = DataHandler.fetch_data(ticker, period, interval)
                if raw_data is not None:
                    st.session_state.raw_data[asset_name] = raw_data
                    st.session_state.processed_data[asset_name] = DataHandler.add_features(
                        raw_data.copy(), ticker)
            
            # Update news and inventory data
            if news_api_key:
                sentiment = news_handler.update_news_sentiment()
                for asset_name in selected_assets:
                    if asset_name in st.session_state.processed_data:
                        st.session_state.processed_data[asset_name]['news_sentiment'] = sentiment
            
            if eia_api_key:
                inventory = econ_handler.update_inventory_data()
                for asset_name in selected_assets:
                    if asset_name in st.session_state.processed_data:
                        st.session_state.processed_data[asset_name]['inventory_level'] = inventory
            
            st.success(f"Data fetched for {len(selected_assets)} assets!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Explorer", 
        "Rule-Based Strategy", 
        "Optimizer",
        "RL Strategy", 
        "Live Trading"
    ])
    
    # Tab 1: Data Explorer
    with tab1:
        st.header("Multi-Asset Data Explorer")
        
        if selected_assets:
            col1, col2 = st.columns(2)
            selected_asset = col1.selectbox("Select Asset to View", selected_assets)
            
            if selected_asset in st.session_state.processed_data:
                df = st.session_state.processed_data[selected_asset]
                
                st.subheader(f"{selected_asset} Price Chart")
                st.line_chart(df['Close'])
                
                st.subheader("Features")
                feature_options = [col for col in df.columns if col not in [
                    'signal', 'position', 'strategy_return', 'cumulative_return'
                ]]
                selected_features = st.multiselect(
                    "Select features to plot",
                    feature_options,
                    default=['volatility_3h', 'volatility_ratio_12h', 'momentum_6h']
                )
                
                if selected_features:
                    st.line_chart(df[selected_features])
                
                st.subheader("Latest Data")
                st.dataframe(df.tail(10))
                
                # LSTM Forecast Training
                st.subheader("LSTM Forecast")
                if st.button("Train LSTM Forecaster"):
                    with st.spinner("Training LSTM model..."):
                        X, y, scaler = DataHandler.prepare_lstm_data(df)
                        if X is not None and y is not None:
                            lstm_forecaster = LSTMForecaster()
                            if lstm_forecaster.train(X, y):
                                st.session_state.lstm_models[selected_asset] = lstm_forecaster
                                st.success("LSTM model trained successfully!")
                                
                                # Make predictions
                                forecasts = lstm_forecaster.predict(X)
                                if forecasts is not None:
                                    # Inverse transform predictions
                                    forecasts = scaler.inverse_transform(forecasts)
                                    # Align with dataframe (shift by lookback)
                                    df_lstm = df.iloc[lstm_forecaster.lookback:].copy()
                                    df_lstm['lstm_forecast'] = forecasts
                                    st.session_state.processed_data[selected_asset] = pd.concat([
                                        df.iloc[:lstm_forecaster.lookback], 
                                        df_lstm
                                    ])
                                    st.line_chart(df_lstm[['log_return', 'lstm_forecast']])
                            else:
                                st.error("LSTM training failed")
                        else:
                            st.error("Not enough data for LSTM training")
                
                # News and sentiment
                if news_api_key and news_handler.cached_news:
                    st.subheader("Latest News & Sentiment")
                    sentiment_col, news_col = st.columns([1, 2])
                    
                    with sentiment_col:
                        current_sentiment = news_handler.cached_sentiment
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=current_sentiment,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Current News Sentiment"},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'steps': [
                                    {'range': [-1, -0.5], 'color': "red"},
                                    {'range': [-0.5, 0], 'color': "orange"},
                                    {'range': [0, 0.5], 'color': "lightgreen"},
                                    {'range': [0.5, 1], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': current_sentiment
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with news_col:
                        for article in news_handler.cached_news[:3]:
                            st.markdown(f"""
                            **{article['title']}**  
                            *{article['source']['name']} - {article['publishedAt']}*  
                            {article['description']}  
                            [Read more]({article['url']})
                            """)
                    
                    # Show sentiment triggers
                    triggers = news_handler.get_recent_triggers()
                    if triggers:
                        st.subheader("Recent Sentiment Triggers")
                        for trigger in triggers[:3]:  # Show most recent 3
                            emoji = "🚀" if trigger['type'] == 'bullish_trigger' else "⚠️"
                            st.markdown(f"""
                            {emoji} **{trigger['type'].replace('_', ' ').title()}**  
                            *{trigger['timestamp']} - Score: {trigger['score']:.2f}*  
                            {trigger['title']}
                            """)
            else:
                st.warning(f"No data available for {selected_asset}. Please fetch data first.")
        else:
            st.warning("No assets selected. Please select assets in the sidebar.")
    
    # Tab 2: Rule-Based Strategy
    with tab2:
        st.header("Rule-Based Trading Strategy")
        
        if selected_assets:
            selected_asset = st.selectbox("Select Asset", selected_assets, key='strategy_asset')
            
            if selected_asset in st.session_state.processed_data:
                df = st.session_state.processed_data[selected_asset]
                
                # Strategy parameters
                col1, col2, col3 = st.columns(3)
                vol_thresh = col1.number_input(
                    "Volatility Threshold", 
                    min_value=0.0001, 
                    max_value=0.1, 
                    value=Config.VOL_THRESH_RANGE[0],
                    step=0.0001,
                    format="%.4f"
                )
                hold_hours = col2.number_input(
                    "Hold Hours", 
                    min_value=1, 
                    max_value=24, 
                    value=6
                )
                use_sentiment = col3.checkbox("Use News Sentiment", value=True)
                
                col4, col5 = st.columns(2)
                sentiment_threshold = col4.slider(
                    "Sentiment Threshold", 
                    min_value=-1.0, 
                    max_value=1.0, 
                    value=-0.5, 
                    step=0.1
                )
                use_lstm = col5.checkbox("Use LSTM Forecast", 
                    value=selected_asset in st.session_state.lstm_models)
                
                # Add sentiment triggers
                use_triggers = st.checkbox("Use Sentiment Triggers", value=True)
                
                if st.button("Run Backtest"):
                    with st.spinner("Running backtest..."):
                        # Update sentiment and inventory data
                        if news_api_key:
                            sentiment = news_handler.update_news_sentiment()
                            df['news_sentiment'] = sentiment
                        
                        if eia_api_key:
                            inventory = econ_handler.update_inventory_data()
                            df['inventory_level'] = inventory
                        
                        # Generate signals
                        signals = TradingStrategy.generate_signals(
                            df,
                            {
                                'vol_thresh': vol_thresh,
                                'hold_hours': hold_hours,
                                'sentiment_thresh': sentiment_threshold
                            },
                            use_sentiment,
                            use_lstm,
                            st.session_state.lstm_models.get(selected_asset)
                        )
                        
                        # Apply sentiment triggers if enabled
                        if use_triggers:
                            triggers = news_handler.get_recent_triggers()
                            signals = TradingStrategy.generate_sentiment_trigger_signals(signals, triggers)
                        
                        st.session_state.backtest_data[selected_asset] = Backtester.run_backtest(
                            signals, Config.TRANSACTION_COST)
                        
                        if st.session_state.backtest_data[selected_asset] is not None:
                            st.success("Backtest completed!")
                
                if selected_asset in st.session_state.backtest_data:
                    # Display metrics
                    metrics = Backtester.calculate_metrics(
                        st.session_state.backtest_data[selected_asset])
                    
                    st.subheader("Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Strategy Return", f"{metrics['strategy_return']*100:.2f}%")
                    col2.metric("Benchmark Return", f"{metrics['benchmark_return']*100:.2f}%")
                    col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Annualized Strategy", f"{metrics['annualized_strategy']*100:.2f}%")
                    col5.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                    col6.metric("Win Rate", f"{metrics['win_rate']*100:.2f}%")
                    
                    # Plot performance
                    st.subheader("Performance Comparison")
                    st.line_chart(st.session_state.backtest_data[selected_asset][
                        ['cumulative_return', 'benchmark_return']])
                    
                    # Plot rolling Sharpe
                    st.subheader("Rolling Sharpe Ratio (6M)")
                    st.line_chart(st.session_state.backtest_data[selected_asset]['rolling_sharpe'])
                    
                    # Show trades
                    st.subheader("Trades")
                    trades = st.session_state.backtest_data[selected_asset][
                        st.session_state.backtest_data[selected_asset]['position'].diff() != 0]
                    st.dataframe(trades[['Close', 'position', 'volatility_3h', 'volatility_ratio_12h']])
            else:
                st.warning(f"No processed data available for {selected_asset}. Please fetch data first.")
        else:
            st.warning("No assets selected. Please select assets in the sidebar.")
    
    # Tab 3: Optimizer
    with tab3:
        st.header("Strategy Optimization")
        
        if selected_assets:
            selected_asset = st.selectbox("Select Asset", selected_assets, key='optimizer_asset')
            
            if selected_asset in st.session_state.processed_data:
                df = st.session_state.processed_data[selected_asset]
                
                st.subheader("Grid Search Parameters")
                col1, col2 = st.columns(2)
                
                # Parameter ranges
                vol_thresh_min = col1.number_input(
                    "Min Volatility Threshold",
                    min_value=0.0001,
                    max_value=0.01,
                    value=Config.VOL_THRESH_RANGE[0],
                    step=0.0001,
                    format="%.4f"
                )
                vol_thresh_max = col2.number_input(
                    "Max Volatility Threshold",
                    min_value=0.0001,
                    max_value=0.01,
                    value=Config.VOL_THRESH_RANGE[1],
                    step=0.0001,
                    format="%.4f"
                )
                vol_thresh_step = st.number_input(
                    "Volatility Threshold Step",
                    min_value=0.0001,
                    max_value=0.001,
                    value=0.0005,
                    step=0.0001,
                    format="%.4f"
                )
                
                hold_hours_min = st.number_input(
                    "Min Hold Hours",
                    min_value=1,
                    max_value=24,
                    value=1
                )
                hold_hours_max = st.number_input(
                    "Max Hold Hours",
                    min_value=1,
                    max_value=24,
                    value=12
                )
                
                # Create parameter grid
                param_grid = {
                    'vol_thresh': np.arange(vol_thresh_min, vol_thresh_max, vol_thresh_step).tolist(),
                    'hold_hours': list(range(hold_hours_min, hold_hours_max + 1)),
                    'target_hour': [9, 10, 11],  # Common trading hours
                    'vol_ratio_thresh': [0.5, 0.6, 0.7, 0.8]
                }
                
                if st.button("Run Optimization"):
                    with st.spinner("Running grid search..."):
                        best_params, results = BacktestOptimizer.grid_search(df, param_grid)
                        
                        if best_params is not None:
                            st.session_state.optimization_results[selected_asset] = {
                                'best_params': best_params,
                                'results': results
                            }
                            st.success("Optimization completed!")
                
                if selected_asset in st.session_state.optimization_results:
                    opt_results = st.session_state.optimization_results[selected_asset]
                    
                    st.subheader("Optimization Results")
                    st.write(f"Best Parameters: {opt_results['best_params']}")
                    
                    # Show top 10 parameter combinations
                    st.subheader("Top Performing Parameter Sets")
                    top_results = opt_results['results'].sort_values(
                        'sharpe', ascending=False).head(10)
                    st.dataframe(top_results)
                    
                    # Visualize parameter performance
                    st.subheader("Parameter Performance")
                    
                    # Volatility threshold vs Sharpe
                    fig, ax = plt.subplots()
                    for hold_hours, group in opt_results['results'].groupby('hold_hours'):
                        ax.plot(
                            group['vol_thresh'], 
                            group['sharpe'], 
                            label=f'Hold {hold_hours}h'
                        )
                    ax.set_xlabel("Volatility Threshold")
                    ax.set_ylabel("Sharpe Ratio")
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.warning(f"No processed data available for {selected_asset}. Please fetch data first.")
        else:
            st.warning("No assets selected. Please select assets in the sidebar.")
    
    # Tab 4: RL Strategy
    with tab4:
        st.header("Reinforcement Learning Strategy")
        
        if selected_assets:
            selected_asset = st.selectbox("Select Asset", selected_assets, key='rl_asset')
            
            if selected_asset in st.session_state.processed_data:
                df = st.session_state.processed_data[selected_asset]
                
                st.subheader("RL Training Configuration")
                col1, col2 = st.columns(2)
                algorithm = col1.selectbox("Algorithm", ['DQN', 'PPO'], index=0)
                timesteps = col2.number_input(
                    "Training Timesteps", 
                    min_value=1000, 
                    max_value=100000, 
                    value=10000, 
                    step=1000
                )
                
                if st.button("Train RL Agent"):
                    with st.spinner("Training RL agent..."):
                        # Update sentiment and inventory data
                        if news_api_key:
                            sentiment = news_handler.update_news_sentiment()
                            df['news_sentiment'] = sentiment
                        
                        if eia_api_key:
                            inventory = econ_handler.update_inventory_data()
                            df['inventory_level'] = inventory
                        
                        # Train model
                        st.session_state.rl_model[selected_asset] = RLTrainer().train_model(
                            df,
                            timesteps,
                            algorithm
                        )
                        
                        # Evaluate model
                        if st.session_state.rl_model[selected_asset] is not None:
                            st.session_state.rl_results[selected_asset] = RLTrainer().evaluate_model(
                                st.session_state.rl_model[selected_asset],
                                df
                            )
                            st.success("Training completed!")
                
                # Show training status
                if selected_asset in st.session_state.rl_model:
                    st.subheader("RL Strategy Performance")
                    
                    if selected_asset in st.session_state.rl_results:
                        results = st.session_state.rl_results[selected_asset]
                        
                        col1, col2 = st.columns(2)
                        col1.metric(
                            "Final Balance", 
                            f"${results['final_balance']:,.2f}",
                            f"{results['return']*100:.2f}%"
                        )
                        col2.metric(
                            "Number of Trades",
                            len(results['trades'])
                        )
                        
                        # Plot portfolio value
                        st.subheader("Portfolio Value")
                        st.line_chart(pd.Series(results['portfolio']))
                        
                        # Show trades
                        if results['trades']:
                            st.subheader("RL Trades")
                            trades_df = pd.DataFrame(results['trades'])
                            st.dataframe(trades_df)
                        else:
                            st.info("No trades were executed by the RL agent.")
            else:
                st.warning(f"No processed data available for {selected_asset}. Please fetch data first.")
        else:
            st.warning("No assets selected. Please select assets in the sidebar.")
    
    # Tab 5: Live Trading
    with tab5:
        st.header("Live Trading Simulation")
        
        if selected_assets:
            selected_asset = st.selectbox("Select Asset", selected_assets, key='live_asset')
            
            st.warning("Live trading requires real-time data connections and brokerage API integration.")
            st.info("""
            **Planned Live Trading Features:**
            - Real-time price streaming
            - Automated trade execution
            - Live performance monitoring
            - Risk management controls
            - Email/SMS alerts for sentiment triggers
            """)
            
            # Placeholder for live trading controls
            if st.checkbox("Enable Live Trading Simulation (Demo Mode)"):
                st.warning("Demo mode only - no real trades will be executed")
                
                # Display live sentiment triggers
                triggers = news_handler.get_recent_triggers()
                if triggers:
                    st.subheader("Active Sentiment Triggers")
                    for trigger in triggers:
                        emoji = "🟢" if trigger['type'] == 'bullish_trigger' else "🔴"
                        st.write(f"""
                        {emoji} **{trigger['type'].replace('_', ' ').title()}**  
                        *{trigger['timestamp']} - Score: {trigger['score']:.2f}*  
                        {trigger['title']}
                        """)
                
                # Placeholder for live trading dashboard
                placeholder = st.empty()
                for i in range(5):
                    placeholder.write(f"Simulating live trading updates... {i+1}/5")
                    time.sleep(1)
                placeholder.success("Demo complete! Real implementation would connect to live data.")
        else:
            st.warning("No assets selected. Please select assets in the sidebar.")

if __name__ == "__main__":
    main()
