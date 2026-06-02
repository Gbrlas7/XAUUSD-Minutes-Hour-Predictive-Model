import pandas as pd
import numpy as np

# In the CSV, it has 3 level headlines, so we just use the second header and change it's column name
df = pd.read_csv("high_Frequency_Gold_Vola tility_2026.csv", header=1) # skipped the 1st header(index 0) and start from 2nd header
df = df.rename(columns={
    'Ticker': 'Datetime', 'GC=F': 'Close', 'GC=F.1': 'High', 
    'GC=F.2': 'Low', 'GC=F.3': 'Open', 'GC=F.4': 'Volume'
})
df = df.iloc[1:].reset_index(drop=True) # drop the empty "Datetime" row

# 2. Format columns
df['Datetime'] = pd.to_datetime(df['Datetime']) # convert string to special panda 'Datetime' objects
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[numeric_cols] = df[numeric_cols].astype(float) # convert string to float
df = df.sort_values('Datetime').reset_index(drop=True) # making sure the data is in order(in terms of time))

# 3. Create Basic Features
df['Return_1m'] = df['Close'].pct_change() # generate return(1 min) in percentage
df['Rolling_Vol_15m'] = df['Return_1m'].rolling(window=15).std() # groups the current minute+14 minutes after it. and calculate the std

# 4. Create the Target (Predicting the return 15 minutes into the future)
# shift the close price 15 steps backwards to align future prices with current rows
df['Future_Close_15m'] = df['Close'].shift(-15)
df['Target_15m_Return'] = (df['Future_Close_15m'] - df['Close']) / df['Close'] # this will be in percentage

# 5. Create a classification target (1 if return > 0.01%, 0 otherwise)
df['Target_Class'] = np.where(df['Target_15m_Return'] > 0.0001, 1, 0)

# Drop NaN(Not a Number) values created by shifting and rolling
df = df.dropna()

print(df[['Datetime', 'Close', 'Target_15m_Return', 'Target_Class']].head(10))