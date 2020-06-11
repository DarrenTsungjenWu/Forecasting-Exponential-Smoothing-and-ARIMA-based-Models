# -*- coding: utf-8 -*-

from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import metrics
from statsmodels.tsa.api import ExponentialSmoothing

EAFV = read_excel('EAFVdata_31324878.xls', sheet_name = 'EAFV', header = 0, index_col = 0, parse_dates = True, squeeze = True)
EAFV_ffit = ExponentialSmoothing(EAFV, seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()
EAFV_FF = EAFV_ffit.forecast(12).rename('Model 1 Forecasting Future 12 Months')
error = EAFV - EAFV_ffit.fittedvalues
mse = sum(error**2)/len(error)
ci_up = EAFV_FF + 1.96*mse
ci_low = EAFV_FF - 1.96*mse
ci_up.plot(ls = '--', color = 'red', label = '95% upper confidence interval', legend = True)
ci_low.plot(ls = '--', color = 'red', label = '95% lower confidence interval', legend = True)
EAFV.plot(color = 'black', label = 'Original Data', legend = True)
EAFV_FF.plot(color = 'red', legend = True)
pyplot.show()

#Residual Plot
resd = EAFV_ffit.resid
plot_acf(resd, lags = 50, title = 'ACF of EAFV Residual')
plot_pacf(resd, lags = 50, title = 'PACF of EAFV Residual')
