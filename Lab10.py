import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. Load dataset
df = pd.read_csv("AirPassengers.csv")
df.rename(columns={"#Passengers": "Passengers"}, inplace=True)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")

ts = df["Passengers"]

# 2. Train–test split (last 12 months)
train = ts[:-12]
test  = ts[-12:]

# 3. Initialize ARIMA(p,d,q) model → ARIMA(2,1,1)
model = ARIMA(train, order=(2, 1, 1))

# 4. Fit the model
model_fit = model.fit()

# 5. Forecast using predict()
forecast = model_fit.predict(start=test.index[0], end=test.index[-1])

print("\nForecasted Values:")
print(forecast)

# 6. Plot results
plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Actual")
plt.plot(forecast, label="ARIMA(2,1,1) Forecast", linestyle="--")
plt.title("ARIMA Forecast vs Actual – AirPassengers")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
