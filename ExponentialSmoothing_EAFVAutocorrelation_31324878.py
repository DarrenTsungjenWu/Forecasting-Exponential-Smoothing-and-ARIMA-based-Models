# -*- coding: utf-8 -*-


#EAFV Autocorrelation
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import metrics

EAFV = read_excel('EAFVdata_31324878.xls', sheet_name = 'EAFV', header = 0, index_col = 0, parse_dates = True, squeeze = True)

###1. Preliminary tasks
#Check the time series plot
EAFV.plot()
pyplot.show()

#Check autocorrelation
autocorrelation_plot(EAFV)
pyplot.show()
#Trend: Yes; Seasonality: Yes
plot_acf(EAFV, lags = 50)
plot_pacf(EAFV, lags = 50)
pyplot.show()