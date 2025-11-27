import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA   # ARMA = ARIMA with d=0

# 1. Load data
df = pd.read_csv("AirPassengers.csv")
df.rename(columns={"#Passengers": "Passengers"}, inplace=True)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
ts = df["Passengers"]

# 2. Train–test split (last 12 as test)
train = ts[:-12]
test  = ts[-12:]

# 3. Initialize ARMA model (ARMA(2,1) → ARIMA(2,0,1))
model = ARIMA(train, order=(2, 0, 1))

# 4. Fit the model
model_fit = model.fit()

# 5. Forecast for the test period using predict()
arma_forecast = model_fit.predict(start=test.index[0], end=test.index[-1])

print("\nForecasted values:")
print(arma_forecast)

# 6. Plot train, test and forecast
plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test (Actual)")
plt.plot(arma_forecast, label="ARMA(2,1) Forecast", linestyle="--")
plt.title("ARMA(2,1) Forecast vs Actual – AirPassengers")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
