# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 02:48:23 2020

@author: Microsoft Windows
"""

#K226
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

K226 = read_excel('K226data_31324878.xls', sheet_name = 'K226', index_col =0, header = 0, parse_dates = True, squeeze = True)

#Model: Holt Linear
from statsmodels.tsa.api import Holt
K226_train = K226[0:210]
K226_test = K226[210:300]

#Model 1: Additive trend
K226_fit0 = Holt(K226_train).fit(optimized=True)
K226_F0 = K226_fit0.forecast(len(K226_test)).rename('Model 1')
#K226_fit0.fittedvalues.plot(color = 'blue')
K226_F0.plot(color = 'blue', legend = True)

K226_fit1 = Holt(K226_train, exponential = True).fit(optimized=True)
K226_F1 = K226_fit1.forecast(len(K226_test)).rename('Model 2')
#K226_fit1.fittedvalues.plot(color = 'red')
K226_F1.plot(color = 'red', legend = True)

#Use Holt-Winter
from statsmodels.tsa.api import ExponentialSmoothing
K226_fit2 = ExponentialSmoothing(K226_train, seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()
K226_F2 = K226_fit2.forecast(len(K226_test)).rename('Model 3')
#K226_fit2.fittedvalues.plot(color = 'green')
K226_F2.plot(color = 'yellow', legend = True)

K226_fit3 = ExponentialSmoothing(K226_train, seasonal_periods = 12, trend = 'mul', seasonal = 'add').fit()
K226_F3 = K226_fit3.forecast(len(K226_test)).rename('Model 4')
#K226_fit3.fittedvalues.plot(color = 'yellow')
K226_F3.plot(color = 'green', legend = True)

K226_fit4 = ExponentialSmoothing(K226_train, seasonal_periods = 12, trend = 'mul', seasonal = 'mul').fit()
K226_F4 = K226_fit4.forecast(len(K226_test)).rename('Model 5')
#K226_fit4.fittedvalues.plot(color = 'orange')
K226_F4.plot(color = 'orange', legend = True)

K226_fit5 = ExponentialSmoothing(K226_train, seasonal_periods = 12, trend = 'add', seasonal = 'mul').fit()
K226_F5 = K226_fit5.forecast(len(K226_test)).rename('Model 6')
#K226_fit5.fittedvalues.plot(color = 'pink')
K226_F5.plot(color = 'pink', legend = True)

K226.plot(color = 'black', label = 'Original Data', legend = True)
pyplot.show()

import numpy as np
from sklearn import metrics
def mape(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 100
def errors(true, pred):
    MSE = metrics.mean_squared_error(true, pred)
    MAE = metrics.mean_absolute_error(true, pred)
    MAPE = mape(true, pred)
    return MSE, MAE, MAPE


#Various errors in training set and test set 
K226_train_e1 = errors(K226_train, K226_fit0.fittedvalues)
K226_train_e2 = errors(K226_train, K226_fit1.fittedvalues)
K226_train_e3 = errors(K226_train, K226_fit2.fittedvalues)
K226_train_e4 = errors(K226_train, K226_fit3.fittedvalues)
K226_train_e5 = errors(K226_train, K226_fit4.fittedvalues)
K226_train_e6 = errors(K226_train, K226_fit5.fittedvalues)
print('Training Set of Model 1: MSE, MAE and MAPE are {0}'.format(K226_train_e1))
print('Training Set of Model 2: MSE, MAE and MAPE are {0}'.format(K226_train_e2))
print('Training Set of Model 3 MSE, MAE and MAPE are {0}'.format(K226_train_e3))
print('Training Set of Model 4: MSE, MAE and MAPE are {0}'.format(K226_train_e4))
print('Training Set of Model 5: MSE, MAE and MAPE are {0}'.format(K226_train_e5))
print('Training Set of Model 6: MSE, MAE and MAPE are {0}'.format(K226_train_e6))
K226_test_e1 = errors(K226_test, K226_F0)
K226_test_e2 = errors(K226_test, K226_F1)
K226_test_e3 = errors(K226_test, K226_F2)
K226_test_e4 = errors(K226_test, K226_F3)
K226_test_e5 = errors(K226_test, K226_F4)
K226_test_e6 = errors(K226_test, K226_F5)
print('Test Set of Model 1: MSE, MAE and MAPE are {0}'.format(K226_test_e1))
print('Test Set of Model 2: MSE, MAE and MAPE are {0}'.format(K226_test_e2))
print('Test Set of Model 3: MSE, MAE and MAPE are {0}'.format(K226_test_e3))
print('Test Set of Model 4: MSE, MAE and MAPE are {0}'.format(K226_test_e4))
print('Test Set of Model 5: MSE, MAE and MAPE are {0}'.format(K226_test_e5))
print('Test Set of Model 6: MSE, MAE and MAPE are {0}'.format(K226_test_e6))



