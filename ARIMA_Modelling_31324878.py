# -*- coding: utf-8 -*-

#SARIMA Modelling
from pandas import read_excel
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  
#plt.style.use('fivethirtyeight')

K54D = read_excel('K54Ddata_31324878.xls', sheet_name = 'K54D', index_col = 0, header = 0, parse_dates = True, squeeze = True)
K54D_train = K54D[0:169]; K54D_test = K54D[169:241]

#Model fitting and its result
model = sm.tsa.statespace.SARIMAX(K54D, order=(0,1,2), seasonal_order=(2,1,1,12)) ###Adjust Parameters and compare their AIC and Q 
SARIMA_fit = model.fit(disp = False)
print(SARIMA_fit.summary())
SARIMA_fit.plot_diagnostics(figsize=(8, 8))
plt.show()

#Make preidction on test set starting with '2013-11-01' (Starting point of test set)
SARIMA_pred = SARIMA_fit.get_prediction(start = pd.to_datetime('2014-02-01'), dynamics = False) #pd.to_datetime('2013-11-01')
test_pred = SARIMA_pred.predicted_mean

#Measure Error
import numpy as np
from sklearn import metrics
def mape(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 100
def errors(true, pred):
    MSE = metrics.mean_squared_error(true, pred)
    MAE = metrics.mean_absolute_error(true, pred)
    MAPE = mape(true, pred)
    return MSE, MAE, MAPE
###Adjust the model and check the behaviour of errors by comparing the change of Q and AIC as well.
SARIMA_train_er = errors(K54D_train, SARIMA_fit.fittedvalues[0:169]) #fit
SARIMA_test_er = errors(K54D_test, SARIMA_pred.predicted_mean) #test_pred
print('The Training Erros in MSE, MAE and MAPE are {0}'.format(SARIMA_train_er))
print('The Test Erros in MSE, MAE and MAPE are {0}'.format(SARIMA_test_er))


###Below is important for visualising the results but can probabily be removed from part of report.
pred_ci = SARIMA_pred.conf_int(alpha = .05) 
#print(pred_ci) >> Returns: Lower printing and upper printing (the interval)

#Plot thier original data and predicted results with confidence interval
ax = K54D['2000':].plot(label='Original data')
SARIMA_pred.predicted_mean.plot(ax=ax, label='Prediction on Test Set', alpha=.5)  

ax.fill_between(pred_ci.index, #Date time index: 1969-01-01, 1969-02-01,...
                pred_ci.iloc[:, 0], #Lower printing
                pred_ci.iloc[:, 1], color='k', alpha=.2) #Upper printing; alpha for the transparency of the interval
plt.legend()
plt.show()

# Forecast future 12 months
SARIMA_fcast = SARIMA_fit.get_forecast(steps=20)
# Obtain 95% confidence intervals
fcast_ci = SARIMA_fcast.conf_int(alpha = .05)

# plotting forecasts ahead
ax = K54D.plot(label='Original data')
SARIMA_fcast.predicted_mean.plot(ax=ax, label='Forecasting Result', 
                                 title='SARIMA Forecasting Future 12 Months with 95% Confidence Interval')
ax.fill_between(fcast_ci.index,
                fcast_ci.iloc[:, 0],
                fcast_ci.iloc[:, 1], color='k', alpha=.25)
plt.legend()
plt.show()


