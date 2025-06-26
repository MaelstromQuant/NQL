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

# Configuration
class Config:
    DEFAULT_TICKER = "CL=F"  # WTI Crude Oil Futures
    DEFAULT_PERIOD = "6mo"
    DEFAULT_INTERVAL = "1h"
    VOL_THRESH = 0.0025  # Default volatility threshold
    HOLD_HOURS = 6  # Default hold period for trades
    TRANSACTION_COST = 0.0005  # 0.05% transaction cost
    NEWS_UPDATE_MINUTES = 15  # Update news every 15 minutes
    EIA_UPDATE_HOURS = 24  # Update inventory data every 24 hours

# Data Handler
class DataHandler:
    @staticmethod
    def fetch_data(ticker, period, interval):
        """Fetch price data from Yahoo Finance"""
        try:
            data = yf.download(tickers=ticker, period=period, interval=interval)
            if data.empty:
                st.error("No data returned. Please check your parameters.")
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    @staticmethod
    def add_features(df):
        """Add technical features to the dataframe"""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Calculate returns
        df['return'] = df['Close'].pct_change()
        
        # Calculate volatility (std of returns)
        df['volatility_3h'] = df['return'].rolling(3).std()
        df['volatility_12h'] = df['return'].rolling(12).std()
        df['volatility_ratio'] = df['volatility_3h'] / df['volatility_12h']
        
        # Calculate momentum
        df['momentum_3h'] = df['Close'].pct_change(3)
        df['momentum_12h'] = df['Close'].pct_change(12)
        
        # Add session labels (Asia, Europe, US)
        df['hour'] = df.index.hour
        df['session'] = 'Other'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'Asia'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'Europe'
        df.loc[(df['hour'] >= 16) & (df['hour'] < 24), 'session'] = 'US'
        
        return df

# News Handler
class NewsHandler:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_update = None
        self.cached_sentiment = 0
        self.cached_news = []
        
    def fetch_latest_headlines(self, query="crude oil OR WTI", page_size=10):
        """Fetch news headlines from NewsAPI"""
        if not self.api_key:
            return []
            
        try:
            url = f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&sortBy=publishedAt&apiKey={self.api_key}"
            response = requests.get(url)
            data = response.json()
            articles = data.get('articles', [])
            return articles
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return []
    
    def update_news_sentiment(self):
        """Update news sentiment if enough time has passed"""
        if not self.api_key:
            return 0
            
        now = datetime.now()
        if (self.last_update is None or 
            (now - self.last_update).total_seconds() > Config.NEWS_UPDATE_MINUTES * 60):
            
            articles = self.fetch_latest_headlines()
            if articles:
                self.cached_news = articles
                scores = [self.analyzer.polarity_scores(article['title'])['compound'] 
                         for article in articles if article.get('title')]
                if scores:
                    self.cached_sentiment = np.mean(scores)
                self.last_update = now
                
        return self.cached_sentiment

# Economic Data Handler
class EconDataHandler:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.last_update = None
        self.cached_inventory = None
        
    def fetch_latest_inventory(self):
        """Fetch latest crude oil inventory data from EIA"""
        if not self.api_key:
            return None
            
        try:
            url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key={self.api_key}&frequency=weekly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=1"
            response = requests.get(url)
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                return data['data'][0]['value']
            return None
        except Exception as e:
            st.error(f"Error fetching inventory data: {e}")
            return None
    
    def update_inventory_data(self):
        """Update inventory data if enough time has passed"""
        if not self.api_key:
            return None
            
        now = datetime.now()
        if (self.last_update is None or 
            (now - self.last_update).total_seconds() > Config.EIA_UPDATE_HOURS * 3600):
            
            inventory = self.fetch_latest_inventory()
            if inventory is not None:
                self.cached_inventory = inventory
                self.last_update = now
                
        return self.cached_inventory

# Trading Strategy
class TradingStrategy:
    @staticmethod
    def generate_signals(df, vol_thresh, hold_hours, use_sentiment=False, sentiment_threshold=-0.5):
        """Generate trading signals based on volatility and optional sentiment"""
        if df is None or df.empty:
            return None
            
        df = df.copy()
        df['signal'] = 0
        
        # Base condition: volatility spike during target hour
        target_hour = 10  # 10 AM (can be parameterized)
        base_condition = (
            (df['hour'] == target_hour) & 
            (df['volatility_3h'] > vol_thresh) & 
            (df['volatility_ratio'] > 0.7
        ))
        
        # Optional sentiment filter
        if use_sentiment and 'news_sentiment' in df:
            sentiment_condition = (df['news_sentiment'] > sentiment_threshold)
            entry_condition = base_condition & sentiment_condition
        else:
            entry_condition = base_condition
        
        # Generate signals
        df.loc[entry_condition, 'signal'] = 1
        
        # Implement hold period
        for i in df[df['signal'] == 1].index:
            end_idx = min(len(df), df.index.get_loc(i) + hold_hours)
            df.loc[i:df.index[end_idx-1], 'position'] = 1
        
        # Fill remaining positions with 0
        df['position'] = df['position'].fillna(0)
        
        return df

# Backtester
class Backtester:
    @staticmethod
    def run_backtest(df, transaction_cost=0.0005):
        """Run backtest on strategy"""
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
        
        return df
    
    @staticmethod
    def calculate_metrics(df):
        """Calculate performance metrics"""
        if df is None or 'strategy_return' not in df:
            return {}
            
        metrics = {}
        
        # Total returns
        metrics['strategy_return'] = df['cumulative_return'].iloc[-1] - 1
        metrics['benchmark_return'] = df['benchmark_return'].iloc[-1] - 1
        
        # Annualized returns
        days = (df.index[-1] - df.index[0]).days
        metrics['annualized_strategy'] = (1 + metrics['strategy_return'])**(365/days) - 1
        metrics['annualized_benchmark'] = (1 + metrics['benchmark_return'])**(365/days) - 1
        
        # Volatility
        metrics['strategy_volatility'] = df['strategy_return'].std() * np.sqrt(252)
        metrics['benchmark_volatility'] = df['return'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        metrics['sharpe_ratio'] = metrics['annualized_strategy'] / metrics['strategy_volatility']
        
        # Max drawdown
        cum_returns = df['cumulative_return']
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        wins = df[df['strategy_return'] > 0]['strategy_return'].count()
        trades = df[df['strategy_return'] != 0]['strategy_return'].count()
        metrics['win_rate'] = wins / trades if trades > 0 else 0
        
        return metrics

# RL Trading Environment
class OilTradingEnv(Env):
    def __init__(self, df, initial_balance=100000, transaction_cost=0.0005):
        super().__init__()
        self.df = df
        self.current_step = 0
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # State space: past returns, volatility, momentum, sentiment, inventory
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(15,), dtype=np.float32
        )
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 10  # Start with enough history
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state vector"""
        # Past 10 returns
        returns = self.df['return'].iloc[self.current_step-10:self.current_step].values
        
        # Current features
        features = [
            self.df['volatility_3h'].iloc[self.current_step],
            self.df['volatility_ratio'].iloc[self.current_step],
            self.df['momentum_3h'].iloc[self.current_step],
            self.df['momentum_12h'].iloc[self.current_step],
            self.df.get('news_sentiment', 0).iloc[self.current_step],
            self.df.get('inventory_level', 0).iloc[self.current_step],
            self.position
        ]
        
        state = np.concatenate([returns, features]).astype(np.float32)
        return state
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0, True, {}
            
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0
        info = {}
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.transaction_cost * self.balance
                info['action'] = 'buy'
        elif action == 2:  # Sell
            if self.position == 1:
                # Calculate profit
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
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Get new state
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")

# RL Agent Trainer
class RLTrainer:
    def __init__(self):
        self.model = None
        self.training = False
        self.progress = 0
    
    def train_model(self, df, total_timesteps=10000, algorithm='DQN'):
        """Train RL model on given data"""
        if df is None or len(df) < 100:
            st.error("Not enough data for training")
            return None
            
        self.training = True
        self.progress = 0
        
        # Create environment
        env = OilTradingEnv(df)
        
        # Create model
        if algorithm == 'DQN':
            model = DQN('MlpPolicy', env, verbose=0)
        else:
            model = PPO('MlpPolicy', env, verbose=0)
        
        # Train in chunks to update progress
        chunk_size = total_timesteps // 10
        for i in range(10):
            model.learn(chunk_size, reset_num_timesteps=False)
            self.progress = (i + 1) * 10
            time.sleep(0.1)  # Simulate training time
            
        self.training = False
        return model
    
    def evaluate_model(self, model, df):
        """Evaluate trained model on data"""
        if model is None or df is None:
            return None
            
        env = OilTradingEnv(df)
        obs = env.reset()
        done = False
        portfolio_values = [env.balance]
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            portfolio_values.append(env.balance)
        
        return {
            'portfolio': portfolio_values,
            'trades': env.trades,
            'final_balance': env.balance,
            'return': (env.balance - env.initial_balance) / env.initial_balance
        }

# Streamlit App
def main():
    st.set_page_config(
        page_title="Crude Oil Trading Dashboard",
        page_icon="🛢️",
        layout="wide"
    )
    
    st.title("🛢️ Crude Oil Trading Dashboard")
    st.markdown("""
    A reinforcement learning-powered dashboard for trading WTI Crude Oil futures.
    Combines technical indicators, news sentiment, and inventory data.
    """)
    
    # Initialize session state
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'backtest_data' not in st.session_state:
        st.session_state.backtest_data = None
    if 'rl_model' not in st.session_state:
        st.session_state.rl_model = None
    if 'rl_results' not in st.session_state:
        st.session_state.rl_results = None
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API keys
    st.sidebar.subheader("API Keys")
    news_api_key = st.sidebar.text_input("NewsAPI Key", type="password")
    eia_api_key = st.sidebar.text_input("EIA API Key", type="password")
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    ticker = st.sidebar.text_input("Ticker Symbol", Config.DEFAULT_TICKER)
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
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    vol_thresh = st.sidebar.number_input(
        "Volatility Threshold", 
        min_value=0.0001, 
        max_value=0.1, 
        value=Config.VOL_THRESH,
        step=0.0001,
        format="%.4f"
    )
    hold_hours = st.sidebar.number_input(
        "Hold Hours", 
        min_value=1, 
        max_value=24, 
        value=Config.HOLD_HOURS
    )
    use_sentiment = st.sidebar.checkbox("Use News Sentiment Filter", value=True)
    sentiment_threshold = st.sidebar.slider(
        "Sentiment Threshold", 
        min_value=-1.0, 
        max_value=1.0, 
        value=-0.5, 
        step=0.1
    )
    
    # Initialize handlers
    news_handler = NewsHandler(news_api_key)
    econ_handler = EconDataHandler(eia_api_key)
    rl_trainer = RLTrainer()
    
    # Data fetching and processing
    st.sidebar.subheader("Data Actions")
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            st.session_state.raw_data = DataHandler.fetch_data(ticker, period, interval)
            if st.session_state.raw_data is not None:
                st.session_state.processed_data = DataHandler.add_features(st.session_state.raw_data)
                
                # Update news sentiment and inventory data
                if news_api_key:
                    sentiment = news_handler.update_news_sentiment()
                    if 'news_sentiment' not in st.session_state.processed_data:
                        st.session_state.processed_data['news_sentiment'] = sentiment
                
                if eia_api_key:
                    inventory = econ_handler.update_inventory_data()
                    if 'inventory_level' not in st.session_state.processed_data:
                        st.session_state.processed_data['inventory_level'] = inventory
                
                st.success("Data fetched and processed successfully!")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Explorer", "Rule-Based Strategy", "RL Strategy", "Live Trading"])
    
    # Tab 1: Data Explorer
    with tab1:
        st.header("Data Explorer")
        
        if st.session_state.raw_data is not None:
            st.subheader("Price Data")
            st.line_chart(st.session_state.raw_data['Close'])
            
            st.subheader("Features")
            feature_cols = ['volatility_3h', 'volatility_12h', 'volatility_ratio', 
                           'momentum_3h', 'momentum_12h', 'news_sentiment']
            available_features = [f for f in feature_cols if f in st.session_state.processed_data.columns]
            st.line_chart(st.session_state.processed_data[available_features])
            
            st.subheader("Latest Data")
            st.dataframe(st.session_state.processed_data.tail(10))
            
            # Show latest news if available
            if news_api_key and news_handler.cached_news:
                st.subheader("Latest News")
                for article in news_handler.cached_news[:5]:
                    st.markdown(f"""
                    **{article['title']}**  
                    *{article['source']['name']} - {article['publishedAt']}*  
                    {article['description']}  
                    [Read more]({article['url']})
                    """)
                    
                # Show sentiment gauge
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
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning("No data available. Please fetch data first.")
    
    # Tab 2: Rule-Based Strategy
    with tab2:
        st.header("Rule-Based Trading Strategy")
        
        if st.session_state.processed_data is not None:
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    # Update sentiment and inventory data
                    if news_api_key:
                        sentiment = news_handler.update_news_sentiment()
                        st.session_state.processed_data['news_sentiment'] = sentiment
                    
                    if eia_api_key:
                        inventory = econ_handler.update_inventory_data()
                        st.session_state.processed_data['inventory_level'] = inventory
                    
                    # Generate signals and run backtest
                    signals = TradingStrategy.generate_signals(
                        st.session_state.processed_data,
                        vol_thresh,
                        hold_hours,
                        use_sentiment,
                        sentiment_threshold
                    )
                    
                    st.session_state.backtest_data = Backtester.run_backtest(signals, Config.TRANSACTION_COST)
                    
                    if st.session_state.backtest_data is not None:
                        st.success("Backtest completed!")
            
            if st.session_state.backtest_data is not None:
                # Display metrics
                metrics = Backtester.calculate_metrics(st.session_state.backtest_data)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Strategy Return", f"{metrics['strategy_return']*100:.2f}%")
                col2.metric("Benchmark Return", f"{metrics['benchmark_return']*100:.2f}%")
                col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Annualized Strategy", f"{metrics['annualized_strategy']*100:.2f}%")
                col5.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                col6.metric("Win Rate", f"{metrics['win_rate']*100:.2f}%")
                
                # Plot performance
                st.subheader("Performance")
                st.line_chart(st.session_state.backtest_data[['cumulative_return', 'benchmark_return']])
                
                # Plot drawdown
                cum_returns = st.session_state.backtest_data['cumulative_return']
                peak = cum_returns.cummax()
                drawdown = (cum_returns - peak) / peak
                st.subheader("Drawdown")
                st.line_chart(drawdown)
                
                # Show trades
                st.subheader("Trades")
                trades = st.session_state.backtest_data[st.session_state.backtest_data['position'].diff() != 0]
                st.dataframe(trades[['Close', 'position', 'volatility_3h', 'volatility_ratio']])
        else:
            st.warning("No processed data available. Please fetch data first.")
    
    # Tab 3: RL Strategy
    with tab3:
        st.header("Reinforcement Learning Strategy")
        
        if st.session_state.processed_data is not None:
            st.subheader("RL Training")
            
            col1, col2 = st.columns(2)
            algorithm = col1.selectbox("Algorithm", ['DQN', 'PPO'], index=0)
            timesteps = col2.number_input("Training Timesteps", min_value=1000, max_value=100000, value=10000, step=1000)
            
            if st.button("Train RL Agent"):
                with st.spinner("Training RL agent..."):
                    # Update sentiment and inventory data
                    if news_api_key:
                        sentiment = news_handler.update_news_sentiment()
                        st.session_state.processed_data['news_sentiment'] = sentiment
                    
                    if eia_api_key:
                        inventory = econ_handler.update_inventory_data()
                        st.session_state.processed_data['inventory_level'] = inventory
                    
                    # Train model
                    st.session_state.rl_model = rl_trainer.train_model(
                        st.session_state.processed_data,
                        timesteps,
                        algorithm
                    )
                    
                    # Evaluate model
                    if st.session_state.rl_model is not None:
                        st.session_state.rl_results = rl_trainer.evaluate_model(
                            st.session_state.rl_model,
                            st.session_state.processed_data
                        )
                        st.success("Training completed!")
            
            # Show training progress
            if rl_trainer.training:
                st.progress(rl_trainer.progress)
                st.write(f"Training in progress... {rl_trainer.progress}% complete")
            
            # Show results
            if st.session_state.rl_results is not None:
                st.subheader("RL Strategy Performance")
                
                col1, col2 = st.columns(2)
                col1.metric("Final Balance", f"${st.session_state.rl_results['final_balance']:,.2f}")
                col2.metric("Total Return", f"{st.session_state.rl_results['return']*100:.2f}%")
                
                # Plot portfolio value
                st.subheader("Portfolio Value")
                st.line_chart(pd.Series(st.session_state.rl_results['portfolio']))
                
                # Show trades
                if st.session_state.rl_results['trades']:
                    st.subheader("RL Trades")
                    trades_df = pd.DataFrame(st.session_state.rl_results['trades'])
                    st.dataframe(trades_df)
                else:
                    st.info("No trades were executed by the RL agent.")
        else:
            st.warning("No processed data available. Please fetch data first.")
    
    # Tab 4: Live Trading
    with tab4:
        st.header("Live Trading Simulation")
        st.warning("This feature is under development and will be implemented in a future update.")
        st.info("""
        Planned features:
        - Real-time data streaming
        - Live trade execution
        - Performance monitoring
        - Risk management controls
        """)

if __name__ == "__main__":
    main()
