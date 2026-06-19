import pandas as pd
import numpy as np

# In the CSV, it has 3 level headlines, so we just use the second header and change it's column name
df = pd.read_csv("high_Frequency_Gold_Vola tility_2026.csv", header=1) # skipped the 1st header(index 0) and start from 2nd header
df = df.rename(columns={
    'Ticker': 'Datetime', 'GC=F': 'Close', 'GC=F.1': 'High', 
    'GC=F.2': 'Low', 'GC=F.3': 'Open', 'GC=F.4': 'Volume'
})
df = df.iloc[1:].reset_index(drop=True) # drop the empty "Datetime" headline row

# 2. Format columns
df['Datetime'] = pd.to_datetime(df['Datetime']) # convert string to special panda 'Datetime' objects
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[numeric_cols] = df[numeric_cols].astype(float) # convert string to float
df = df.sort_values('Datetime') # making sure the data is in order(in terms of time))
df.set_index('Datetime', inplace=True) # set as index


# 3. Create the Target (Predicting the return 15 minutes into the future)
# shift the close price 15 steps backwards(upwards) to align future prices with current rows
df['Future_Close_15m'] = df['Close'].shift(-15)
df['Target_15m_Return'] = (df['Future_Close_15m'] - df['Close']) / df['Close'] # this will be in percentage

# 4. Create a classification target (1 if return > 0.01%, 0 otherwise)
df['Target_Class'] = np.where(df['Target_15m_Return'] > 0.0001, 1, 0)





# --- Momentum ---
windows = [1, 3, 5, 15, 60]
for w in windows:
    df[f'Return_{w}m'] = df['Close'].pct_change(periods=w) # generate return(1,3,5,15,60 min) in percentage

# --- Volatility ---
df['Rolling_Vol_15m'] = df['Return_1m'].rolling(window=15).std() # groups the current minute and 14 minutes before it and gives its std
df['Rolling_Vol_60m'] = df['Return_1m'].rolling(window=60).std()

high_low = df['High'] - df['Low'] # calculates the distance between high and low in the current minute
high_close = np.abs(df['High'] - df['Close'].shift(1)) # These calculate the absolute distance (np.abs()) from the previous minute's closing price to the current minute's high and low.
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1) # puts those three distances side-by-side and uses .max(axis=1) to pick the biggest for every single row
#Average True Range (ATR)
df['ATR_14'] = true_range.rolling(window=14).mean() # mean of the last 14 min true range
df['NATR_14'] = df['ATR_14'] / df['Close']

# --- Technical Oscillators ---
# RSI (Relative Strength Index)
# RSI looks at the size of recent gains versus recent losses over a set period (usually 14 periods) and converts that into a score from 0 to 100
delta = df['Close'].diff() # calculates diff between previous row from current row value
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=7).mean()
avg_loss = loss.rolling(window=7).mean()
rs = avg_gain / (avg_loss + 1e-8)
df['RSI_7'] = 100 - (100 / (1 + rs))

# Moving Average
# looks at the relationship between two moving averages: a "fast" short-term average (12 periods) and a "slow" long-term average (26 periods)
# creates a rolling window that assigns exponentially decreasing weights to older prices
df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean() 
# distance to EMA (Stationary Mean-Reversion Feature)
# Positive = Price is above the EMA. Negative = Price is below the EMA.
df['Dist_To_EMA_9'] = (df['Close'] - df['EMA_9']) / df['EMA_9']
df['Dist_To_EMA_50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']

# Positive = Fast EMA is above Slow EMA (Bullish Trend)
# Negative = Fast EMA is below Slow EMA (Bearish Trend)
df['EMA_9_vs_21'] = (df['EMA_9'] - df['EMA_21']) / df['EMA_21']

# Bollinger Bands
bb_mean = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
df['BB_Upper'] = bb_mean + (bb_std * 2)
df['BB_Lower'] = bb_mean - (bb_std * 2)
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)

# Stochastic
roll_high = df['High'].rolling(window=14).max()
roll_low = df['Low'].rolling(window=14).min()
df['Stoch_K'] = 100 * ((df['Close'] - roll_low) / (roll_high - roll_low + 1e-8))
df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
df['Stoch_K_vs_D'] = df['Stoch_K'] - df['Stoch_D']

# Microstructure / Volume
# Relative Volume (RVOL)
df['Vol_vs_60m_Avg'] = df['Volume'] / (df['Volume'].rolling(window=60).mean() + 1e-8)
# Price Volume Trends (take accounts not just absolute volume but volume from buy or sell)
df['Price_Vol_Trend'] = np.sign(df['Return_1m']) * df['Vol_vs_60m_Avg']

# VWAP
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['Vol_x_Price'] = df['Typical_Price'] * df['Volume']
df['Cum_Vol_Day'] = df.groupby(df.index.date)['Volume'].cumsum()
df['Cum_Vol_x_Price_Day'] = df.groupby(df.index.date)['Vol_x_Price'].cumsum()
df['VWAP'] = df['Cum_Vol_x_Price_Day'] / (df['Cum_Vol_Day'] + 1e-8)
# Positive = Price is above the institutional average (Overbought)
# Negative = Price is below the institutional average (Oversold)
df['Dist_To_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']




# Support and Resistance
df['Resistance_60'] = df['High'].shift(1).rolling(window=60).max()
df['Support_60'] = df['Low'].shift(1).rolling(window=60).min()
# Now, if Close breaks ABOVE Resistance, the distance becomes NEGATIVE
df['Dist_To_Resistance_60'] = (df['Resistance_60'] - df['Close']) / df['Close']
df['Dist_To_Support_60'] = (df['Close'] - df['Support_60']) / df['Close']

# Dow Theory
df['Prev_Resistance_60'] = df['High'].shift(61).rolling(window=60).max()
df['Prev_Support_60'] = df['Low'].shift(61).rolling(window=60).min()
# Positive = Higher Highs (Dow Uptrend). Negative = Lower Highs (Dow Downtrend).
df['Dow_High_Structure'] = (df['Resistance_60'] - df['Prev_Resistance_60']) / df['Prev_Resistance_60']
# Positive = Higher Lows (Dow Uptrend). Negative = Lower Lows (Dow Downtrend).
df['Dow_Low_Structure'] = (df['Support_60'] - df['Prev_Support_60']) / df['Prev_Support_60']

# Elliott Wave Theory
# define the macro swing high and swing low (e.g., last 120 minutes)
macro_window = 120
df['Macro_High_120'] = df['High'].rolling(window=macro_window).max()
df['Macro_Low_120'] = df['Low'].rolling(window=macro_window).min()
macro_range = df['Macro_High_120'] - df['Macro_Low_120']
# calculate retracement depth
# 0.0 means no pullback, 1.0 means 100% retracement
# 0.618 means price has pulled back exactly 61.8% from the high (Golden Ratio)
df['Fibonacci_Retracement'] = (df['Macro_High_120'] - df['Close']) / (macro_range + 1e-8)
# Is the price currently resting right in the "Elliott Wave Bounce Zone" (between 50% and 61.8%)?
is_in_golden_pocket = (df['Fibonacci_Retracement'] > 0.50) & (df['Fibonacci_Retracement'] < 0.62)
df['Pattern_Fib_Golden_Pocket'] = is_in_golden_pocket.astype(int)

# Wyckoff Method
# Effort vs. Result
candle_body_size = np.abs(df['Close'] - df['Open']) / df['Close']
# divide effort (relative volume) by result (candle size)
# very high number means massive volume but the price went nowhere (Absorption)
df['Wyckoff_Effort_Result'] = df['Vol_vs_60m_Avg'] / (candle_body_size + 1e-8)
# The "Spring"
# Rule 1: The absolute Low of the minute broke below the 50-minute floor
went_below_support = df['Low'] < df['Support_60']
# Rule 2: But the minute Closed back safely ABOVE the floor (Retail was trapped)
closed_above_support = df['Close'] > df['Support_60']
# Rule 3: High volume during this
high_volume_trap = df['Vol_vs_60m_Avg'] > 1.2
df['Pattern_Wyckoff_Spring'] = (went_below_support & closed_above_support & high_volume_trap).astype(int)
# The "Upthrust" (inverse of the Spring)
went_above_resistance = df['High'] > df['Resistance_60']
closed_below_resistance = df['Close'] < df['Resistance_60']
df['Pattern_Wyckoff_Upthrust'] = (went_above_resistance & closed_below_resistance & high_volume_trap).astype(int)





# CHART PATTERNS

# The Double Top
# Definition: Two consecutive peaks are almost exactly the same height, 
# and the price just broke below the local support (the neckline).
is_equal_peaks = df['Dow_High_Structure'].abs() < 0.0005
is_neckline_broken = df['Dist_To_Support_60'] < 0
# Convert boolean (True/False) to integers (1/0) for XGBoost
df['Pattern_Double_Top'] = (is_equal_peaks & is_neckline_broken).astype(int)

# The Double Bottom
# Definition: Two consecutive floors are almost exactly the same depth,
# and the price just broke above the local resistance (the neckline).
is_equal_floors = df['Dow_Low_Structure'].abs() < 0.0005
is_resistance_broken = df['Dist_To_Resistance_60'] < 0
df['Pattern_Double_Bottom'] = (is_equal_floors & is_resistance_broken).astype(int)

# Head & Shoulders
# Resistance_60 = Right Shoulder | Prev_Resistance_60 = Head | Prev_Prev = Left Shoulder
df['Prev_Prev_Resistance_60'] = df['High'].shift(121).rolling(window=60).max()
is_head_taller_than_left = df['Prev_Resistance_60'] > df['Prev_Prev_Resistance_60']
is_right_lower_than_head = df['Resistance_60'] < df['Prev_Resistance_60']

is_shoulders_even = ((df['Resistance_60'] - df['Prev_Prev_Resistance_60']).abs() / df['Prev_Prev_Resistance_60']) < 0.002
is_neckline_broken = df['Dist_To_Support_60'] < 0
is_hs_setup = is_head_taller_than_left & is_right_lower_than_head & is_shoulders_even
df['Pattern_Head_Shoulders'] = (is_hs_setup & is_neckline_broken).astype(int)

# Inverse Head & Shoulders
df['Prev_Prev_Support_60'] = df['Low'].shift(121).rolling(window=60).min()
is_head_deeper_than_left = df['Prev_Support_60'] < df['Prev_Prev_Support_60']
is_right_higher_than_head = df['Support_60'] > df['Prev_Support_60']

is_inv_shoulders_even = ((df['Support_60'] - df['Prev_Prev_Support_60']).abs() / df['Prev_Prev_Support_60']) < 0.002
is_inv_neckline_broken = df['Dist_To_Resistance_60'] < 0
is_inv_hs_setup = is_head_deeper_than_left & is_right_higher_than_head & is_inv_shoulders_even
df['Pattern_Inv_Head_Shoulders'] = (is_inv_hs_setup & is_inv_neckline_broken).astype(int)

# 3. The Symmetrical Triangle / Wedge Squeeze (Continuation)
# Definition: Volatility has completely compressed to its lowest levels.
# measure this by checking if the current Bollinger Band width is small
bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
is_volatility_crushed = bb_width < 0.002
# We only trigger the pattern if the volatility is crushed AND volume suddenly spikes
is_volume_surging = df['Vol_vs_60m_Avg'] > 1.5
is_breakout_event = is_volatility_crushed & is_volume_surging

is_breaking_up = df['Return_1m'] > 0
is_breaking_down = df['Return_1m'] < 0
df['Pattern_Wedge_Up'] = (is_breakout_event & is_breaking_up).astype(int)
df['Pattern_Wedge_Down'] = (is_breakout_event & is_breaking_down).astype(int)

# CLEANUP
# drop all NaNs created by the 15m target shift and the 60m historical rolling windows
df.dropna(inplace=True)
# drop Future_Close column so the model can't cheat by looking at it
df.drop(columns=['Future_Close_15m'], inplace=True)


print("\nSample Features:")
print(df[['Close', 'Target_Class', 'Return_60m', 'RSI_7', 'BB_Position']].head())
cols_to_drop = ['Resistance_60', 'Support_60', 'High', 'Low', 'Open', 'Close', 'BB_Upper', 'BB_Lower', 'EMA_9', 'EMA_21', 'EMA_50',
                'ATR_14', 'Volume', 'Typical_Price', 'Vol_x_Price', 'Cum_Vol_Day', 'Cum_Vol_x_Price_Day', 'VWAP', 'Prev_Resistance_60', 
                'Prev_Support_60', 'Prev_Prev_Resistance_60', 'Prev_Prev_Support_60', 'Macro_High_120', 'Macro_Low_120']
df.drop(columns=cols_to_drop, inplace=True)




import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

X = df.drop(columns=['Target_15m_Return', 'Target_Class'])
y = df['Target_Class']

# chronological train/test split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]


# create a blank baseline model
base_model = xgb.XGBClassifier(
    random_state=42
)

# define the Grid (the settings we want to test)
# max_depth: How many consecutive IF/THEN rules the tree can make
# learning_rate: How aggressively it corrects its mistakes
# subsample: What % of the data it looks at per tree (prevents overfitting)
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'n_estimators': [100]
}

# initialize grid search
# cv=3 means it will test each combination 3 times to ensure it wasn't a fluke
# scoring='precision' forces it to strictly optimize for high-quality Buy signals
grid_search = GridSearchCV(
    estimator=base_model, 
    param_grid=param_grid, 
    scoring='precision', 
    cv=3, 
    verbose=1
)

# running the search
grid_search.fit(X_train, y_train)

# find the ultimate model
print(f"Best Settings Found: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

# making predicition using the best model
y_pred = best_model.predict(X_test)

# evaluation of the best model
print("\n--- Optimized Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

importance = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importance.sort_values(ascending=False).head(5)
print("\n--- Top 5 Most Important Features ---")
print(top_features)
