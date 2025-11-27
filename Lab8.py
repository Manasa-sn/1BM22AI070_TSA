import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load data
df = pd.read_csv("AirPassengers.csv")
df.rename(columns={"#Passengers": "Passengers"}, inplace=True)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
ts = df["Passengers"]

# 2. Plot ACF & PACF
plt.figure(figsize=(10,4))
plot_acf(ts, lags=40)
plt.title("ACF")
plt.show()

plt.figure(figsize=(10,4))
plot_pacf(ts, lags=40, method="ywm")
plt.title("PACF")
plt.show()

# 3. Train–test split
train = ts[:-12]
test  = ts[-12:]

# Evaluation helper
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# 4. MA(1) model
# ARIMA (p, q, r): p -- AR, q -- Differencing, r -- MA
ma1 = ARIMA(train, order=(0,0,1)).fit()
ma1_pred = ma1.forecast(steps=12)
ma1_mae, ma1_mse, ma1_rmse = metrics(test, ma1_pred)

print("\nMA(1) Performance:")
print(f"MAE  = {ma1_mae:.3f}")
print(f"MSE  = {ma1_mse:.3f}")
print(f"RMSE = {ma1_rmse:.3f}")

# 5. Try higher MA orders
results = []
for q in [2, 3, 5, 10]:
    try:
        model = ARIMA(train, order=(0,0,q)).fit()
        pred = model.forecast(steps=12)
        mae, mse, rmse = metrics(test, pred)
        results.append([q, mae, mse, rmse])
    except:
        pass

results_df = pd.DataFrame(results, columns=["q", "MAE", "MSE", "RMSE"])
print("\nMA(q) Comparison:")
print(results_df)

# 6. Plot best model
best_q = results_df.sort_values("RMSE").iloc[0]["q"]
best_model = ARIMA(train, order=(0,0,int(best_q))).fit()
best_pred = best_model.forecast(steps=12)

plt.figure(figsize=(10,5))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(best_pred, label=f"Best MA({int(best_q)}) Forecast", linestyle="--")
plt.title(f"Best MA Model (MA({int(best_q)}))")
plt.legend()
plt.grid(True)
plt.show()
