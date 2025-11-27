import yfinance as yf
from datetime import datetime
import pandas as pd

tesla = yf.Ticker("TSLA")
start_time = datetime(2020, 1, 1)
end_time = datetime(2025, 1, 1)
#for all the periods
df_tsla_all = tesla.history(period = 'max')
#For the specified time period
df_tsla = tesla.history(start = start_time, end = end_time)

#To convert the time to a specifies format YYYY-MM-DD
df_tsla['Date'] = pd.to_datetime(df_tsla['Date'], format = '%Y-%M-%d')
df_tsla.set_index('Date', inplace = True)
df_tsla.head()

df_tsla['Close'].plot(xlim = ['2024-01-01', '2025-01-01'], title = 'Tesla Stocks', c = 'green', ls = '--')
#c - color, ls - line style

#Resampling the data
df_tsla['Close'].resample('ME').mean().plot(title = 'Monthly Average Closing Price')
#ME - Month End

df_tsla['Close'].resample('YE').mean().plot(title = 'Yearly Average Closing Price', ls = '--')

df_tsla['Close'].resample('QE').mean().plot(title = 'Quarterly Average Closing Price')
#Quarterly end

df_tsla['Close'].resample('YE').mean().plot(kind = 'bar', title = 'Yearly Average Closing Price')
