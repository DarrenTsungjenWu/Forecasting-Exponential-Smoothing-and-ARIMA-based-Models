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


#Model: Holt-Winter
from statsmodels.tsa.api import ExponentialSmoothing
K54D_train = K54D[0:169]; K54D_test = K54D[169:241]

#Model 1
K54D_fit1 = ExponentialSmoothing(K54D_train.astype(float), seasonal_periods = 12, trend = 'add', seasonal = 'mul').fit()
K54D_F1 = K54D_fit1.forecast(len(K54D_test)).rename('Model 1')
#K54D_fit1.fittedvalues.plot(color = 'red')
K54D_F1.plot(color = 'red', legend = True)

#Model 2
K54D_fit2 = ExponentialSmoothing(K54D_train.astype(float), seasonal_periods = 12, trend = 'mul', seasonal = 'mul').fit()
K54D_F2 = K54D_fit2.forecast(len(K54D_test)).rename('Model 2')
#K54D_fit2.fittedvalues.plot(color = 'blue')
K54D_F2.plot(color = 'blue', legend = True)

#Model 3
K54D_fit3 = ExponentialSmoothing(K54D_train.astype(float), seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()
K54D_F3 = K54D_fit1.forecast(len(K54D_test)).rename('Model 3')
#K54D_fit3.fittedvalues.plot(color = 'red')
K54D_F3.plot(color = 'green', legend = True)

#Model 4
K54D_fit4 = ExponentialSmoothing(K54D_train.astype(float), seasonal_periods = 12, trend = 'mul', seasonal = 'add').fit()
K54D_F4 = K54D_fit2.forecast(len(K54D_test)).rename('Model 4')
#K54D_fit4.fittedvalues.plot(color = 'blue')
K54D_F4.plot(color = 'pink', legend = True)


K54D.plot(color = 'black', label = 'Original Data', legend = True)
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

K54D_train_e1 = errors(K54D_train, K54D_fit1.fittedvalues)
K54D_train_e2 = errors(K54D_train, K54D_fit2.fittedvalues)
K54D_train_e3 = errors(K54D_train, K54D_fit3.fittedvalues)
K54D_train_e4 = errors(K54D_train, K54D_fit4.fittedvalues)
print('Training Set of Model 1: MSE, MAE and MAPE are {0}'.format(K54D_train_e1))
print('Training Set of Model 2: MSE, MAE and MAPE are {0}'.format(K54D_train_e2))
print('Training Set of Model 3: MSE, MAE and MAPE are {0}'.format(K54D_train_e3))
print('Training Set of Model 4: MSE, MAE and MAPE are {0}'.format(K54D_train_e4))
K54D_test_e1 = errors(K54D_test, K54D_F1)
K54D_test_e2 = errors(K54D_test, K54D_F2)
K54D_test_e3 = errors(K54D_test, K54D_F3)
K54D_test_e4 = errors(K54D_test, K54D_F4)
print('Test Set of Model 1: MSE, MAE and MAPE are {0}'.format(K54D_test_e1))
print('Test Set of Model 2: MSE, MAE and MAPE are {0}'.format(K54D_test_e2))
print('Test Set of Model 3: MSE, MAE and MAPE are {0}'.format(K54D_test_e3))
print('Test Set of Model 4: MSE, MAE and MAPE are {0}'.format(K54D_test_e4))


