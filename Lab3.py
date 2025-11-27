import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = {
    'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'value': [np.nan, 2, np.nan, 4, np.nan, 6, 7, np.nan, 9, 10]
}

df = pd.DataFrame(data)

print("Missing values:\n", df.isnull().sum())

df['value'] = df['value'].interpolate(method='linear')

# EXTRA STEP — Fill any leftover NaN just in case
df['value'] = df['value'].fillna(method='bfill')
df['value'] = df['value'].fillna(method='ffill')


print("\nAfter filling:\n", df)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

plt.figure(figsize=(10,4))
plt.plot(df['value'])
plt.title("Time Series")
plt.grid(True)
plt.show()

result = seasonal_decompose(df['value'], model='additive', period=2)

result.plot()
plt.show()
