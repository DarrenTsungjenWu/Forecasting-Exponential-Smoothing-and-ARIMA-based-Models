# -*- coding: utf-8 -*-



#Differencing and Model Selection(testing)
import statsmodels.api as sm  
from pandas import read_excel
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('fivethirtyeight')

K54D = read_excel('K54Ddata_31324878.xls', sheet_name = 'K54D', index_col = 0, header = 0, parse_dates = True, squeeze = True)
#Detrend: 1st Differencing
dat = K54D.values
first_diff = list()
for i in range(1, len(dat)):
    first_values = dat[i] - dat[i - 1]
    first_diff.append(first_values)
#Plot seasonally differenced series
plt.plot(first_diff)
plt.title("1st Differenced Time Plot of K54D")
plt.show()
plot_acf(first_diff, lags = 50, title = 'ACF of 1st Differencing')
plt.show();plot_pacf(first_diff, lags = 50, title = 'PACF of 1st Differencing')
plt.show()
#Note: Trend seemingly disappeared!! But clearly, the seasonality is till there.

#Do Seasonal Differencing on previous results------1st+Seasonal Differencing------to remove seasonality
first_season_diff = list()
for i in range(12, len(first_diff)):
    values = first_diff[i] - first_diff[i - 12]
    first_season_diff.append(values)
plt.plot(first_season_diff)
plt.title("1st + Seasonal Differenced Time Plot of K54D")
plt.show()

plot_acf(first_season_diff, lags = 50, title = 'ACF of 1st + Seasonal Differencing')
plot_pacf(first_season_diff, lags = 50, title = 'PACF of 1st + Seasonal Differencing')