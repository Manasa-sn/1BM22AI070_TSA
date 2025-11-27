import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/bitcoin_price.csv')
df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%M-%d')
df.set_index('Date', inplace = True)
df.head()
df['Close'].plot(xlim = ['2020-01-01', '2022-01-01'])

df['Close'].resample('YE').mean().plot()

df['Close'].resample('ME').mean().idxmax()


data = pd.DataFrame({
    'Date': pd.date_range(start = '2023-01-01', periods = 10, freq = 'D'),
    'value': [np.nan, 2, np.nan, 4, np.nan, 6,7, np.nan, 9, 10]
})

data.set_index('Date', inplace = True)
dfs = data.interpolate(method = 'linear')
print("Linear Interpolation")
print(dfs)

df = pd.read_csv('/content/bitcoin_price.csv', index_col = 'Date', parse_dates = True)
df.reset_index('Date', inplace = True)
df.set_index('Date', inplace = True)

df.isnull().sum()
