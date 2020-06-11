# -*- coding: utf-8 -*-


#K226
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

K226 = read_excel('K226data_31324878.xls', sheet_name = 'K226', index_col =0, header = 0, parse_dates = True, squeeze = True)
K226_train = K226[0:210]
K226_test = K226[210:300]
#Check time plot
K226.plot()
pyplot.show()
#The series has trend but has ambiguous seasonality

#Check autocorrelation
autocorrelation_plot(K226, label = 'Autocorrelation Plot of K226')
pyplot.show()

plot_acf(K226, lags = 50)
plot_pacf(K226, lags = 50)
pyplot.show()
