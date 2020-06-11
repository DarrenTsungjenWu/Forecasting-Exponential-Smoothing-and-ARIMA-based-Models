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
#Model 1
JQ2J_fit1 = ExponentialSmoothing(JQ2J_train, seasonal_periods = 12, trend = 'mul', seasonal = 'mul').fit()
JQ2J_F1 = JQ2J_fit1.forecast(len(JQ2J_test)).rename('Model 1')
#JQ2J_fit1.fittedvalues.plot(color = 'red')
JQ2J_F1.plot(color = 'red', legend = True)
#pyplot.show() 

#Model 2
JQ2J_fit2 = ExponentialSmoothing(JQ2J_train, seasonal_periods = 12, trend = 'add', seasonal = 'mul').fit()
JQ2J_F2 = JQ2J_fit2.forecast(len(JQ2J_test)).rename('Model 2')
#JQ2J_fit2.fittedvalues.plot(color = 'green')
JQ2J_F2.plot(color = 'green', legend = True)
#pyplot.show() 

#Model 3
JQ2J_fit3 = ExponentialSmoothing(JQ2J_train, seasonal_periods = 12, trend = 'mul', seasonal = 'add').fit()
JQ2J_F3 = JQ2J_fit3.forecast(len(JQ2J_test)).rename('Model 3')
#JQ2J_fit3.fittedvalues.plot(color = 'pink')
JQ2J_F3.plot(color = 'pink', legend = True)
#pyplot.show() 

#Model 4
JQ2J_fit4 = ExponentialSmoothing(JQ2J_train, seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()
JQ2J_F4 = JQ2J_fit2.forecast(len(JQ2J_test)).rename('Model 4')
#JQ2J_fit4.fittedvalues.plot(color = 'blue')
JQ2J_F4.plot(color = 'blue', legend = True)
#pyplot.show() 


#Plot the fitted value, forecasting reuslts and original series data altogether
#Plot original data
JQ2J.plot(color = 'black', label = 'Original Data', legend = True)
pyplot.show() 

#Various Errors of Training Set and Test set
#Define a MAPE function

import numpy as np
from sklearn import metrics
def mape(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 100
def errors(true, pred):
    MSE = metrics.mean_squared_error(true, pred)
    MAE = metrics.mean_absolute_error(true, pred)
    MAPE = mape(true, pred)
    return MSE, MAE, MAPE

JQ2J_train_e1 = errors(JQ2J_train, JQ2J_fit1.fittedvalues)
JQ2J_train_e2 = errors(JQ2J_train, JQ2J_fit2.fittedvalues)
JQ2J_train_e3 = errors(JQ2J_train, JQ2J_fit3.fittedvalues)
JQ2J_train_e4 = errors(JQ2J_train, JQ2J_fit4.fittedvalues)
print('Training Set of Model 1: MSE, MAE and MAPE are {0}'.format(JQ2J_train_e1))
print('Training Set of Model 2: MSE, MAE and MAPE are {0}'.format(JQ2J_train_e2))
print('Training Set of Model 3: MSE, MAE and MAPE are {0}'.format(JQ2J_train_e3))
print('Training Set of Model 4: MSE, MAE and MAPE are {0}'.format(JQ2J_train_e4))
JQ2J_test_e1 = errors(JQ2J_test, JQ2J_F1)
JQ2J_test_e2 = errors(JQ2J_test, JQ2J_F2)
JQ2J_test_e3 = errors(JQ2J_test, JQ2J_F3)
JQ2J_test_e4 = errors(JQ2J_test, JQ2J_F4)
print('Test Set of Model 1: MSE, MAE and MAPE are {0}'.format(JQ2J_test_e1))
print('Test Set of Model 2: MSE, MAE and MAPE are {0}'.format(JQ2J_test_e2))
print('Test Set of Model 3: MSE, MAE and MAPE are {0}'.format(JQ2J_test_e3))
print('Test Set of Model 4: MSE, MAE and MAPE are {0}'.format(JQ2J_test_e4))

