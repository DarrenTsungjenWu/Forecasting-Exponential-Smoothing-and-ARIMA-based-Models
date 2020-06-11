# -*- coding: utf-8 -*-


#K54D
import numpy as np
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

K54D = read_excel('K54Ddata_31324878.xls', sheet_name = 'K54D', index_col = 0, header = 0, parse_dates = True, squeeze = True)
#K54D = np.genfromtxt('K54D.csv', delimiter = ',', skip_header = 1)


#Forecasting
K54D_ffit = ExponentialSmoothing(K54D, seasonal_periods = 12, trend = 'add', seasonal = 'mul').fit()
K54D_FF = K54D_ffit.forecast(12).rename('Model 1 Forecasting Future 12 Months')
error = K54D - K54D_ffit.fittedvalues
mse = sum(error**2)/len(error)
ci_up = K54D_FF + 1.96*mse
ci_low = K54D_FF - 1.96*mse
ci_up.plot(ls = '--', color = 'pink', label = '95% upper confidence interval', legend = True)
ci_low.plot(ls = '--', color = 'pink', label = '95% lower confidence interval', legend = True)
K54D.plot(color = 'black', label = 'Original Data', legend = True)
K54D_FF.plot(color = 'pink', legend = True)
pyplot.show()

#Residual Plot
resd = K54D_ffit.resid
plot_acf(resd, lags = 50, title = 'ACF of K54D Residual')
plot_pacf(resd, lags = 50, title = 'PACF of K54D Residual')