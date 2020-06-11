# -*- coding: utf-8 -*-


#JQ2J
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

JQ2J = read_excel('JQ2Jdata_31324878.xls', sheet_name = 'JQ2J', index_col =0, header = 0, parse_dates = True, squeeze = True)
#Check time plot
JQ2J.plot()
pyplot.show()


#Check autocorrelation
autocorrelation_plot(JQ2J, label = 'Autocorrelation Plot of JQ2J')
pyplot.show()

plot_acf(JQ2J, lags = 50)
plot_pacf(JQ2J, lags = 50)
pyplot.show()
