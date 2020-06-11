# -*- coding: utf-8 -*-

#JQ2J
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

JQ2J = read_excel('JQ2Jdata_31324878.xls', sheet_name = 'JQ2J', index_col =0, header = 0, parse_dates = True, squeeze = True)
JQ2J_train = JQ2J[0:170]
JQ2J_test = JQ2J[170:240]

#Holt-Winter
from statsmodels.tsa.api import ExponentialSmoothing

#Forecasting
JQ2J_ffit = ExponentialSmoothing(JQ2J, seasonal_periods = 12, trend = 'mul', seasonal = 'mul').fit()
JQ2J_FF = JQ2J_ffit.forecast(12).rename('Model 1 Forecasting Future 12 Months')
error = JQ2J - JQ2J_ffit.fittedvalues
mse = sum(error**2)/len(error)
ci_up = JQ2J_FF + 1.96*mse
ci_low = JQ2J_FF - 1.96*mse
print(ci_up)
print(ci_low)
#ci_up.plot(ls = '--', color = 'red', label = '95% upper confidence interval', legend = True)
#ci_low.plot(ls = '--', color = 'red', label = '95% upper confidence interval', legend = True)
JQ2J.plot(color = 'black', label = 'Original Data', legend = True)
JQ2J_FF.plot(color = 'red', legend = True)
pyplot.show()

#Residual Plot
resd = JQ2J_ffit.resid
plot_acf(resd, lags = 50, title = 'ACF of JQ2J Residual')
plot_pacf(resd, lags = 50, title = 'PACF of JQ2J Residual')
