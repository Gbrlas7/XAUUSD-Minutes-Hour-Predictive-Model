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

# --- Technical Oscillators ---
# RSI (Relative Strength Index)
# RSI looks at the size of recent gains versus recent losses over a set period (usually 14 periods) and converts that into a score from 0 to 100
delta = df['Close'].diff() # calculates diff between previous row from current row value
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
# looks at the relationship between two moving averages: a "fast" short-term average (12 periods) and a "slow" long-term average (26 periods)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean() # creates a rolling window that assigns exponentially decreasing weights to older prices
ema_26 = df['Close'].ewm(span=26, adjust=False).mean() # span=12 defines the decay rate, and adjust=False tells Pandas to calculate it recursively
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean() # takes a 9-period EMA of the MACD line itself (lag of real-time MACD)

# Bollinger Bands
bb_mean = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
df['BB_Upper'] = bb_mean + (bb_std * 2)
df['BB_Lower'] = bb_mean - (bb_std * 2)
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

# --- Microstructure / Volume ---
# Relative Volume (RVOL)
df['Vol_vs_60m_Avg'] = df['Volume'] / (df['Volume'].rolling(window=60).mean() + 1e-8)
# Price Volume Trends (take accounts not just absolute volume but volume from buy or sell)
df['Price_Vol_Trend'] = np.sign(df['Return_1m']) * df['Volume']

# CLEANUP
# drop all NaNs created by the 15m target shift and the 60m historical rolling windows
df.dropna(inplace=True)
# drop Future_Close column so the model can't cheat by looking at it
df.drop(columns=['Future_Close_15m'], inplace=True)

print(f"Final Dataset Shape: {df.shape}")
print("\nSample Features:")
print(df[['Close', 'Target_Class', 'Return_60m', 'RSI_14', 'BB_Position']].head())
