import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Load data
df = pd.read_csv("AirPassengers.csv")
df.rename(columns={"#Passengers": "Passengers"}, inplace=True)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")

ts = df["Passengers"]

# 2. Moving average (12-month) to detect trend
df["MA_12"] = ts.rolling(window=12).mean()

# Simple numeric trend: fit a straight line to MA_12
ma12 = df["MA_12"].dropna()
x = np.arange(len(ma12))
y = ma12.values
slope, intercept = np.polyfit(x, y, 1)  # fitting a polynomical with degree 1
print(f"Slope of 12-month MA ≈ {slope:.3f} passengers per month")

# 3. Plot original series + 12-month moving average
plt.figure(figsize=(10, 5))
plt.plot(ts, label="Original")
plt.plot(df["MA_12"], label="12-Month MA (Trend)", linewidth=2)
plt.title("AirPassengers with 12-Month Moving Average")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. ACF and PACF
plt.figure(figsize=(10, 4))
plot_acf(ts, lags=40)
plt.title("ACF – AirPassengers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(ts, lags=40, method="ywm")
plt.title("PACF – AirPassengers")
plt.tight_layout()
plt.show()
