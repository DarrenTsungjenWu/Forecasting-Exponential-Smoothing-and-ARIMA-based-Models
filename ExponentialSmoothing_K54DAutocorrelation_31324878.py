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

#Check time series plot
K54D.plot()
pyplot.show()
#Series has both trend and seasonality

#Check autocorrelation
autocorrelation_plot(K54D)
pyplot.show()

plot_acf(K54D, lags = 50)
plot_pacf(K54D, lags = 50)
pyplot.show()