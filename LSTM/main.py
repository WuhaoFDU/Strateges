import tushare as ts
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime
from config import TS_TOKEN, STOCK_CODE, START_DATE, END_DATE
import pandas_ta as ta
from sklearn.utils.class_weight import compute_class_weight

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# Matplotlib Chinese font settings
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Songti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

data = pro.daily(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
data['trade_date'] = pd.to_datetime(data['trade_date'])
data.index = data['trade_date']
data = data.sort_index()
data = data[['close','open', 'high', 'low',  'vol','amount']]

data['ma5'] = ta.sma(data['close'], length=5)
data['ma20'] = ta.sma(data['close'], length=20)
data['ma60'] = ta.sma(data['close'], length=60)
# Add MACD features computed on close; shift by 1 to avoid lookahead
macd_df = ta.macd(data['close'])
macd_cols = list(macd_df.columns)
data = pd.concat([data, macd_df], axis=1)
data[macd_cols] = data[macd_cols].shift(1)
# Add RSI feature; shift by 1 to avoid lookahead
rsi14 = ta.rsi(data['close'], length=14)
data['rsi14'] = rsi14.shift(1)
data = data.dropna()

lookback = 20
split_idx_rows = int(len(data) * 0.5)
train_data = data.iloc[:split_idx_rows].copy()
test_data = data.iloc[split_idx_rows:].copy()

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

def build_sequences_regression(scaled_array: np.ndarray, closes: pd.Series, window: int) -> tuple:
    X_seq, y_seq = [], []
    for i in range(window, len(scaled_array)):
        X_seq.append(scaled_array[i - window:i])
        next_close = closes.iloc[i]
        y_seq.append(next_close)
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = build_sequences_regression(scaled_train, train_data['close'], lookback)
X_test, y_test = build_sequences_regression(scaled_test, test_data['close'], lookback)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=80, batch_size=32, verbose=1, shuffle=False)

pred_close = model.predict(X_test).ravel()
true_close = y_test.ravel()

# previous close aligns with the end of each input window in test segment
test_close_series = test_data['close'].values
prev_close = test_close_series[lookback-1:len(test_close_series)-1]

pred_labels = (pred_close > prev_close).astype(int)
true_labels = (true_close > prev_close).astype(int)
pred_up_days = int(pred_labels.sum())
pred_down_days = int(len(pred_labels) - pred_up_days)
total_correct = int((pred_labels == true_labels).sum())
win_rate = float(total_correct / len(pred_labels)) if len(pred_labels) > 0 else 0.0

print(f"预判涨的天数: {pred_up_days}")
print(f"预判跌的天数: {pred_down_days}")
print(f"胜率: {win_rate:.4f}")

# Plot predicted vs actual close values for the test set
dates = test_data.index[lookback:]
plt.figure(figsize=(12, 5))
plt.plot(dates, true_close, label='实际收盘', linewidth=1.5)
plt.plot(dates, pred_close, label='预测收盘', linewidth=1.2)
plt.title('预测 vs 实际 收盘价')
plt.legend()
plt.tight_layout()
plt.show()
