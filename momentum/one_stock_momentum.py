import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class one_stock_momentum():
    def __init__(self,df,window=3):
        self.data = df
        self.window = window
    def pre_process(self):
        self.data = self.data[['trade_date','close']]
        self.data = self.data.set_index('trade_date')
        self.data['buy_and_hold'] = self.data['close'] / self.data['close'].iloc[0]
        return 0
    def generate_momentum_factor(self):
        df = self.data
        df['ma3'] = df['close'].rolling(self.window).mean()
        df['velocity'] = df['ma3'] - df['ma3'].shift(self.window)
        df['acceleration'] = df['velocity'] - df['velocity'].shift(self.window)
        df.dropna(inplace=True)
        return df
    def generate_trade_signal(self):
        df = self.data
        df['buy'] = (df['velocity'] > 0) & (df['acceleration'] > 0)
        return df
    def buy_strategy(self,take_profit=0.03,stop_loss=0.03):
        df = self.data
        df['strategy_return'] = 0.0
        # win:1  lose:-1
        df['win_or_lose'] = 0
        buy_state = df['buy'].iloc[0]
        buy_price = 0.0
        one_time_return = 0.0
        for i in range(1,len(df)):
            if buy_state:
                one_time_return = df['close'].iloc[i] / buy_price - 1
                if one_time_return > take_profit or one_time_return < -stop_loss:
                    df.loc[df.index[i], 'strategy_return'] = (one_time_return>0)*take_profit+(one_time_return<-stop_loss)*(-stop_loss)
                    buy_state = False
                    df.loc[df.index[i], 'win_or_lose'] = (one_time_return>0)*2-1
            else:
                if df['buy'].iloc[i]:
                    buy_state = True
                    buy_price = df['close'].iloc[i]
        df['accumulate_return'] = (1+df['strategy_return']).cumprod()
        print(df)
        return df
    
    def plot_strategy_comparison(self):
        """绘制策略对比图"""
        df = self.data
        plt.figure(figsize=(14,8))
        plot_data = pd.DataFrame({
            'Date': df.index,
            'Buy and Hold': df['buy_and_hold'],
            'Momentum Strategy': df['accumulate_return']
        })
        plot_data_long = plot_data.melt(id_vars=['Date'], var_name='Strategy', value_name='Performance')
        sns.lineplot(data=plot_data_long, x='Date', y='Performance', hue='Strategy')
        plt.title('Strategy Performance Comparison')
        plt.show()
        
        # 打印策略表现统计
        print(f"Buy and Hold Final Return: {(df['buy_and_hold'].iloc[-1] - 1)*100:.2f}%")
        print(f"Momentum Strategy Final Return: {(df['accumulate_return'].iloc[-1] - 1)*100:.2f}%")
        
        # 计算策略表现差异
        performance_diff = df['accumulate_return'].iloc[-1] - df['buy_and_hold'].iloc[-1]
        print(f"Performance Difference: {performance_diff*100:.2f}%")
        
        return df


# 主程序
data = pd.read_csv("data_of_200_stocks/000001.csv")

stock = one_stock_momentum(data)
stock.pre_process()
stock.generate_momentum_factor()
stock.generate_trade_signal()
stock.buy_strategy()
print(stock.data)

# 绘制策略对比图
stock.plot_strategy_comparison()
