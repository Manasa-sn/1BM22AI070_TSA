import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# 1. Load time series (AirPassengers)
df = pd.read_csv("AirPassengers.csv")
df.rename(columns={"#Passengers": "Passengers"}, inplace=True)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
ts = df["Passengers"]

# 2. Generate white noise (same length as series)
np.random.seed(42)
white_noise = pd.Series(
    np.random.randn(len(ts)),
    index=ts.index,
    name="WhiteNoise"
)

# 3. Plot time series and white noise
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(ts)
plt.title("AirPassengers Time Series")
plt.ylabel("Passengers")

plt.subplot(2, 1, 2)
plt.plot(white_noise)
plt.title("White Noise Series")
plt.ylabel("Value")
plt.xlabel("Year")

plt.tight_layout()
plt.show()

# 4. ADF + KPSS helper
def adf_kpss(series, name):
    print(f"\n=== {name} ===")

    # ADF
    print("\nADF")
    adf_result = adfuller(series.dropna(), autolag="AIC")
    adf_stat = adf_result[0]  # Test-statistic
    adf_p    = adf_result[1]  # p-value
    adf_crit = adf_result[4]  # Critical values
    print(f"ADF statistic: {adf_stat:.4f}, p-value: {adf_p:.4f}")
    print("ADF critical values:", adf_crit)
    if adf_p < 0.05:
      print("→ ADF conclusion: Reject H0, time series is stationary")
    else:
      print("→ ADF conclusion: Fail to reject H0, time series is STATIONARY")

    # KPSS
    print("\nKPSS")
    kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
    # c --> Stationary around a constant mean
    # ct --> Stationary around linear trend
    kpss_stat = kpss_result[0]
    kpss_p    = kpss_result[1]
    kpss_crit = kpss_result[3]
    print(f"KPSS statistic: {kpss_stat:.4f}, p-value: {kpss_p:.4f}")
    print("KPSS critical values:", kpss_crit)
    if kpss_p < 0.05:
      print("→ KPSS conclusion: Time series is NOT stationary")
    else:
      print("→ KPSS conclusion: Time series is STATIONARY")



# 5. Run tests on both series
adf_kpss(ts, "AirPassengers")
adf_kpss(white_noise, "White Noise")
