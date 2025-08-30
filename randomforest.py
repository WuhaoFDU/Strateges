import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from config import TS_TOKEN, STOCK_CODE

ts.set_token(TS_TOKEN)
pro=ts.pro_api()
df = pro.daily(ts_code=STOCK_CODE, 
    start_date='20100101', end_date='20250731'
    ).set_index('trade_date').sort_index()
df['tomorrow_close'] = df['close'].shift(-1)
df.index=pd.to_datetime(df.index)
df['ma5']=df['close'].rolling(5).mean()
df['ma20']=df['close'].rolling(20).mean()
df['ma60']=df['close'].rolling(60).mean()
df = df.dropna()

features = ['close','pct_chg','vol','amount','ma5','ma20','ma60']
X = df[features]
y = df['tomorrow_close']
start_time , end_time = df.index.min() , df.index.max()
mid_time = start_time + (end_time - start_time) / 3 *2
train_X, train_y = X[features][X.index<mid_time],df['tomorrow_close'][df.index<mid_time]
val_X, val_y = X[features][X.index>=mid_time],pd.DataFrame(df['tomorrow_close'][df.index>=mid_time])
forest_model = RandomForestRegressor(random_state=30)
forest_model.fit(train_X,train_y)
pred_y = pd.DataFrame(forest_model.predict(val_X))
pred_y = pred_y.rename(columns={0:"tomorrow_predict"})
pred_y.index = val_y.index

strategy = val_y.join(pred_y)
strategy['close'] = strategy['tomorrow_close'].shift(1)
strategy['buy'] = strategy['tomorrow_predict'] > strategy['close']
strategy['pct_change'] = strategy['close'].pct_change()
strategy['1+pct_change'] = strategy['pct_change']+1
strategy = strategy[['close', 'buy', '1+pct_change']]
strategy['asset_change'] = 1
strategy.dropna(inplace=True)
strategy['asset_change'] = strategy['buy'].shift(1) * strategy['1+pct_change'] + 1 - strategy['buy'].shift(1)
original_price = strategy['close'].iloc[0]
strategy.dropna(inplace=True)
strategy['strategy'] = strategy['asset_change'].cumprod()

plt.figure(figsize=(20,8))
plt.title("Strategy")
sns.lineplot(data=strategy['strategy'],label='strategy')
sns.lineplot(data=strategy['close'] / original_price, label='base')
fixed_profit_rate = strategy['close'].iloc[-1] / original_price
profit_rate = strategy['strategy'].iloc[-1] - 1
alpha = strategy['strategy'].iloc[-1] / fixed_profit_rate - 1
print(f"收益:{profit_rate}")
print(f"超额收益:{alpha}")




