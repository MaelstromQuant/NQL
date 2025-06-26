import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import warnings
from scipy.stats import linregress

# Suppress warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
class Config:
    HOLD_HOURS = 3
    TRANSACTION_COST = 0.0002  # 2 basis points
    VOL_THRESHOLDS = np.linspace(0.1, 0.5, 5)  # More granular thresholds
    HOURS_TO_TEST = list(range(0, 24))
    SYMBOL = 'CL=F'  # WTI Crude Oil
    PERIOD = '180d'  # Extended period for more robust testing
    INTERVAL = '1h'
    RISK_FREE_RATE = 0.0  # For Sharpe ratio calculation
    
    # Exchange session hour ranges in UTC (more precise definitions)
    SESSION_RANGES = {
        'Asia': range(0, 6),      # Tokyo/Singapore/HongKong
        'Europe': range(7, 16),   # London/Frankfurt overlap
        'US': range(13, 21),       # New York
        'Electronic': range(21, 24) + range(0, 1)  # Overnight electronic
    }

# === DATA HANDLING ===
class DataHandler:
    @staticmethod
    def fetch_data(symbol=Config.SYMBOL, period=Config.PERIOD, interval=Config.INTERVAL):
        """Fetch data with error handling and caching"""
        try:
            data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
            if data.empty:
                raise ValueError("No data returned from Yahoo Finance")
            return data.dropna()
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    @staticmethod
    def add_features(df):
        """Enhanced feature engineering"""
        df = df.copy()
        
        # Basic features
        df['return'] = df['Close'].pct_change()
        df['hour'] = df.index.hour
        df['date'] = df.index.date
        
        # Volatility measures
        df['volatility_3h'] = df['Close'].rolling(window=3).std()
        df['volatility_12h'] = df['Close'].rolling(window=12).std()
        df['volatility_ratio'] = df['volatility_3h'] / df['volatility_12h']
        
        # Momentum indicators
        df['momentum_3h'] = df['Close'].pct_change(3)
        df['momentum_12h'] = df['Close'].pct_change(12)
        
        # Session features
        df = DataHandler.label_sessions(df)
        
        return df.dropna()

    @staticmethod
    def label_sessions(df):
        """Label trading sessions with overlap handling"""
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
    def run_backtest(df):
        """Comprehensive backtesting with enhanced metrics"""
        df = df.copy()
        
        # Calculate strategy returns
        df['strategy_return'] = df['position'].shift(1) * df['return']
        df['strategy_return'] -= Config.TRANSACTION_COST * df['position'].diff().abs().fillna(0)
        
        # Cumulative returns
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
        df['buy_hold'] = (1 + df['return']).cumprod()
        
        # Trade logging
        trades = Backtester._log_trades(df)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Performance metrics
        metrics = Backtester._calculate_metrics(df, trades_df)
        
        return df, trades_df, metrics

    @staticmethod
    def _log_trades(df):
        """Detailed trade logging"""
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
                open_trade.update({
                    'exit_time': df.index[i],
                    'exit_price': df['Close'].iloc[i],
                    'exit_session': df['session'].iloc[i],
                    'holding_period': (df.index[i] - open_trade['entry_time']).total_seconds() / 3600,
                    'return': (df['Close'].iloc[i] - open_trade['entry_price']) / open_trade['entry_price'],
                    'return_net': (df['Close'].iloc[i] - open_trade['entry_price']) / open_trade['entry_price'] - 2*Config.TRANSACTION_COST
                })
                trades.append(open_trade)
                open_trade = None
        
        return trades

    @staticmethod
    def _calculate_metrics(df, trades_df):
        """Enhanced performance metrics"""
        if len(trades_df) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trades': 0
            }
        
        # Basic metrics
        total_return = df['cumulative_return'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (365/(len(df)/24)) - 1
        
        # Risk-adjusted metrics
        excess_returns = df['strategy_return'] - Config.RISK_FREE_RATE/24
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(24)
        
        downside_returns = excess_returns[excess_returns < 0]
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(24) if len(downside_returns) > 0 else 0
        
        # Drawdown
        cum_returns = df['cumulative_return']
        max_dd = (cum_returns / cum_returns.cummax() - 1).min()
        
        # Trade metrics
        win_rate = len(trades_df[trades_df['return_net'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
        profit_factor = trades_df[trades_df['return_net'] > 0]['return_net'].sum() / \
                       abs(trades_df[trades_df['return_net'] < 0]['return_net'].sum()) if \
                       len(trades_df[trades_df['return_net'] < 0]) > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': len(trades_df)
        }

# === OPTIMIZATION ===
class Optimizer:
    @staticmethod
    def grid_search(raw_data):
        """Perform parameter optimization with walk-forward validation"""
        results = []
        
        # Split data into training and validation sets
        split_idx = int(len(raw_data) * 0.7)
        train_data = raw_data.iloc[:split_idx]
        valid_data = raw_data.iloc[split_idx:]
        
        # Parameter grid
        param_grid = {
            'hour': Config.HOURS_TO_TEST,
            'vol_thresh': Config.VOL_THRESHOLDS,
            'hold_hours': [2, 3, 4, 6]
        }
        
        # Train phase
        train_results = []
        for hour in param_grid['hour']:
            for vol in param_grid['vol_thresh']:
                for hold in param_grid['hold_hours']:
                    df = DataHandler.add_features(train_data.copy())
                    df = TradingStrategy.generate_signals(df, hour, vol, hold)
                    _, _, metrics = Backtester.run_backtest(df)
                    
                    train_results.append({
                        'hour': hour,
                        'vol_thresh': vol,
                        'hold_hours': hold,
                        'train_sharpe': metrics['sharpe_ratio'],
                        'train_return': metrics['total_return']
                    })
        
        # Get top N configurations from training
        train_df = pd.DataFrame(train_results)
        top_configs = train_df.sort_values('train_sharpe', ascending=False).head(10)
        
        # Validate top configurations
        for _, config in top_configs.iterrows():
            df = DataHandler.add_features(valid_data.copy())
            df = TradingStrategy.generate_signals(df, config['hour'], config['vol_thresh'], config['hold_hours'])
            _, _, metrics = Backtester.run_backtest(df)
            
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

# === VISUALIZATION ===
class Visualizer:
    @staticmethod
    def plot_results(df, trades_df, best_config):
        """Enhanced visualization of results"""
        plt.style.use('seaborn')
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Equity curve
        axes[0].plot(df.index, df['cumulative_return'], label=f"Strategy (hour={best_config['hour']}, vol>{best_config['vol_thresh']:.2f})")
        axes[0].plot(df.index, df['buy_hold'], label='Buy & Hold', alpha=0.7)
        axes[0].set_title('Strategy Performance')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].grid(True)
        axes[0].legend()
        
        # Drawdown
        dd = (df['cumulative_return'] / df['cumulative_return'].cummax() - 1)
        axes[1].fill_between(df.index, dd, 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(True)
        
        # Hourly returns heatmap
        if not trades_df.empty:
            hourly_returns = trades_df.groupby('entry_session')['return_net'].mean()
            hourly_returns.plot(kind='bar', ax=axes[2], color='green', alpha=0.6)
            axes[2].set_title('Average Returns by Trading Session')
            axes[2].set_ylabel('Average Return')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plots
        Visualizer._plot_trade_analysis(trades_df)

    @staticmethod
    def _plot_trade_analysis(trades_df):
        """Plot trade-level analysis"""
        if trades_df.empty:
            return
            
        plt.figure(figsize=(14, 6))
        
        # Trade returns distribution
        plt.subplot(1, 2, 1)
        sns.histplot(trades_df['return_net'], bins=30, kde=True)
        plt.title('Distribution of Trade Returns')
        plt.xlabel('Return')
        plt.grid(True)
        
        # Returns by hour
        plt.subplot(1, 2, 2)
        trades_df['entry_hour'] = trades_df['entry_time'].dt.hour
        hourly_returns = trades_df.groupby('entry_hour')['return_net'].mean()
        hourly_returns.plot(kind='bar', color='blue', alpha=0.6)
        plt.title('Average Returns by Entry Hour')
        plt.xlabel('Hour (UTC)')
        plt.ylabel('Average Return')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# === MAIN EXECUTION ===
def main():
    # Fetch and prepare data
    raw_data = DataHandler.fetch_data()
    if raw_data is None:
        print("Failed to fetch data. Exiting...")
        return
    
    # Optimize strategy
    print("Running optimization...")
    optimized_results = Optimizer.grid_search(raw_data)
    best_config = optimized_results.iloc[0].to_dict()
    
    # Run backtest with best configuration
    df = DataHandler.add_features(raw_data.copy())
    df = TradingStrategy.generate_signals(
        df, 
        best_config['hour'], 
        best_config['vol_thresh'], 
        best_config['hold_hours']
    )
    df, trades_df, metrics = Backtester.run_backtest(df)
    
    # Display results
    print("\n=== BEST STRATEGY CONFIGURATION ===")
    print(f"Entry Hour (UTC): {best_config['hour']}")
    print(f"Volatility Threshold: {best_config['vol_thresh']:.3f}")
    print(f"Holding Period (hours): {best_config['hold_hours']}")
    
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Trades: {metrics['trades']}")
    
    # Visualize results
    Visualizer.plot_results(df, trades_df, best_config)
    
    # Show sample trades
    if not trades_df.empty:
        print("\n=== SAMPLE TRADES ===")
        print(trades_df.head())

if __name__ == "__main__":
    main()
