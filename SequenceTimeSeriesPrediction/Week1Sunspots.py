import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot

df = pd.read_csv("data\Sunspots.csv", parse_dates=["Date"], index_col="Date")
series = df["Monthly Mean Total Sunspot Number"].asfreq("1M")

series.plot(figsize=(12, 5))
plt.show()

series["1995-01-01":].plot()
plt.show()

series.diff(1).plot()
plt.axis([0, 100, -50, 50])
plt.show()

autocorrelation_plot(series)
plt.show()

autocorrelation_plot(series.diff(1)[1:])
plt.show()

autocorrelation_plot(series.diff(1)[1:].diff(11 * 12)[11*12+1:])
plt.axis([0, 500, -0.1, 0.1])
plt.show()

autocorrelation_plot(series.diff(1)[1:])
plt.axis([0, 50, -0.1, 0.1])
plt.show()

series_diff = series
for lag in range(50):
  series_diff = series_diff[1:] - series_diff[:-1]

autocorrelation_plot(series_diff)
plt.show()



series_diff1 = pd.Series(series[1:] - series[:-1])
autocorrs = [series_diff1.autocorr(lag) for lag in range(1, 60)]
plt.plot(autocorrs)
plt.show()