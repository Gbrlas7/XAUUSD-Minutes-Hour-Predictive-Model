import pandas as pd
import numpy as np


# 1. Create a results dataframe aligned with your test data dates
backtest = pd.DataFrame(index=X_test.index)

# 2. Get the actual forward 15-minute return of the asset
backtest['Actual_Return_15m'] = df.loc[X_test.index, 'Target_15m_Return']

# 3. Get the raw probabilities from XGBoost
probabilities = best_model.predict_proba(X_test)
backtest['Prob_Sell'] = probabilities[:, 0]
backtest['Prob_Buy'] = probabilities[:, 1]

# 4. Define Confidence Thresholds & Generate Signals
# 1 = Long (Buy), -1 = Short (Sell), 0 = Cash (Do nothing)
buy_threshold = 0.65
sell_threshold = 0.65
backtest['Signal'] = 0 # Default to holding cash
backtest.loc[backtest['Prob_Buy'] > buy_threshold, 'Signal'] = 1
backtest.loc[backtest['Prob_Sell'] > sell_threshold, 'Signal'] = -1

# 5. Define Market Frictions (Spread + Slippage + Commission)
transaction_cost = 0.0002 

# 6. Calculate Strategy Returns
# We shift the signal by 1 because a signal generated at 10:00 AM 
backtest['Position'] = backtest['Signal'].shift(1)
# A trade occurs when the position changes (e.g., from 0 to 1, or 1 to -1)
backtest['Trade_Executed'] = (backtest['Position'] != backtest['Position'].shift(1)).astype(int)
backtest['Strategy_Return'] = (backtest['Position'] * backtest['Actual_Return_15m']) - (backtest['Trade_Executed'] * transaction_cost)

# 7. Calculate Cumulative Growth
backtest['Market_Cumulative'] = (1 + backtest['Actual_Return_15m']).cumprod()
backtest['Strategy_Cumulative'] = (1 + backtest['Strategy_Return']).cumprod()



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# 1. Calculate Core Performance Metrics
# Assuming 252 trading days a year, and 96 15-minute periods per day (24 hours)
periods_per_year = 252 * 96 

# Total Return
total_strat_return = backtest['Strategy_Cumulative'].iloc[-1] - 1
total_market_return = backtest['Market_Cumulative'].iloc[-1] - 1

# Annualized Volatility (Risk)
strat_volatility = backtest['Strategy_Return'].std() * np.sqrt(periods_per_year)

# Sharpe Ratio (Risk-Adjusted Return, assuming 0% risk-free rate for simplicity)
sharpe_ratio = (backtest['Strategy_Return'].mean() * periods_per_year) / (strat_volatility + 1e-8)

# Win Rate (Only counting periods where a trade was actively held)
active_trades = backtest[backtest['Position'] != 0]
win_rate = len(active_trades[active_trades['Strategy_Return'] > 0]) / (len(active_trades) + 1e-8)

# Maximum Drawdown (The worst peak-to-trough drop)
rolling_max = backtest['Strategy_Cumulative'].cummax()
drawdown = (backtest['Strategy_Cumulative'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# 2. Build the Visualization Dashboard
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('XGBoost 15m Quantitative Strategy Tearsheet', fontsize=18, fontweight='bold')

# --- Panel 1: Cumulative Equity Curve ---
ax1.plot(backtest.index, backtest['Strategy_Cumulative'], label='XGBoost Strategy (Post-Friction)', color='#00ff00', linewidth=2)
ax1.plot(backtest.index, backtest['Market_Cumulative'], label='Buy & Hold Baseline', color='#888888', alpha=0.7, linestyle='--')
ax1.set_title('Cumulative Return vs. Baseline', fontsize=14)
ax1.set_ylabel('Growth of $1')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

metrics_text = (
    f"Total Return: {total_strat_return:.2%}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Win Rate: {win_rate:.2%}\n"
    f"Max Drawdown: {max_drawdown:.2%}\n"
    f"Annual Volatility: {strat_volatility:.2%}"
)
ax1.text(0.02, 0.75, metrics_text, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'))

# --- Panel 2: Underwater Plot (Drawdowns) ---
ax2.fill_between(backtest.index, drawdown, 0, color='#ff0000', alpha=0.5)
ax2.set_title('Underwater Plot (Drawdowns)', fontsize=14)
ax2.set_ylabel('Drawdown %')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()
