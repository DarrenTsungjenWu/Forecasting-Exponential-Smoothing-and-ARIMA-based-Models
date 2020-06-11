# -*- coding: utf-8 -*-

#Regression Preliminary
from pandas import read_excel
import matplotlib.pyplot as plt
import pandas as pd
FTSE_reg = read_excel('FTSEdata_31324878.xls', sheet_name='FTSE_reg', header=0, 
                     squeeze=True)

from pandas import read_excel
from statsmodels.formula.api import ols

#Regression Model
#Basic
FTSE_Y, EAFV, K226, JQ2J, K54D = FTSE_reg.FTSE, FTSE_reg.EAFV, FTSE_reg.K226, FTSE_reg.JQ2J, FTSE_reg.K54D 

#reading the indicator variables
D1 = FTSE_reg.D1
D2 = FTSE_reg.D2
D3 = FTSE_reg.D3
D4 = FTSE_reg.D4
D5 = FTSE_reg.D5
D6 = FTSE_reg.D6
D7 = FTSE_reg.D7
D8 = FTSE_reg.D8
D9 = FTSE_reg.D9
D10 = FTSE_reg.D10
D11 = FTSE_reg.D11
time = FTSE_reg.time

#Potential Models
mod1 = 'FTSE_Y ~ EAFV + K226 + JQ2J + K54D'
mod2 = 'FTSE_Y ~ EAFV + K226 + JQ2J + K54D + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 '
mod3 = 'FTSE_Y ~ EAFV + K226 + JQ2J + K54D + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 +time'

#Ordinary Least Squares (OLS)
result1 = ols(mod1, data = FTSE_reg).fit()
result2 = ols(mod2, data = FTSE_reg).fit()
result3 = ols(mod3, data = FTSE_reg).fit()

# the results from IndividualSignificance.py, 
# StatsWithIndicatorsBank.py, 
# and StatsWithIndicatorsTimeBank.py are summarised 
# for easy comparison of the key statistics 

import numpy as np
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.api import ExponentialSmoothing
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#Fitting and Forecasting of Independent Variables
EAFV_train = EAFV[0:170];EAFV_test = EAFV[170:239]
K226_train = K226[0:170];K226_test = K226[170:239]
JQ2J_train = EAFV[0:170];JQ2J_test = EAFV[170:239]
K54D_train = K54D[0:170];K54D_test = K54D[170:239]

EAFV_fit = ExponentialSmoothing(EAFV_train, seasonal_periods = 12, trend = 'add', seasonal = 'add').fit() 
EAFV_Fcast = EAFV_fit.forecast(len(EAFV_test))
K226_fit = ExponentialSmoothing(K226_train, seasonal_periods = 12, trend = 'mul', seasonal = 'add').fit()
K226_Fcast = K226_fit.forecast(len(K226_test))
JQ2J_fit = ExponentialSmoothing(JQ2J_train, seasonal_periods = 12, trend = 'mul', seasonal = 'mul').fit()
JQ2J_Fcast = JQ2J_fit.forecast(len(JQ2J_test))
K54D_fit = ExponentialSmoothing(K54D_train.astype(float), seasonal_periods = 12, trend = 'mul', seasonal = 'add').fit()
K54D_Fcast = K54D_fit.forecast(len(K54D_test))

###Comparison of 3 models
#First Model
mod1 = 'FTSE_Y ~ EAFV + K226 + JQ2J + K54D'
mod1_result = ols(mod1, data = FTSE_reg).fit()
#mod1_result.summary(); 
b0 = mod1_result.params.Intercept;b1 = mod1_result.params.EAFV;b2 = mod1_result.params.K226;b3 = mod1_result.params.JQ2J
b4 = mod1_result.params.K54D
#The array of 4 fitted results of independent variables

a1 = np.array(EAFV_fit.fittedvalues);a2 = np.array(K226_fit.fittedvalues);
a3 = np.array(JQ2J_fit.fittedvalues);a4 = np.array(K54D_fit.fittedvalues)
aa1 = np.array(EAFV_fit.fittedvalues); aa2 = np.array(K226_fit.fittedvalues);
aa3 = np.array(JQ2J_fit.fittedvalues); aa4 = np.array(K54D_fit.fittedvalues)

aaa1 = np.array(EAFV_fit.fittedvalues); aaa2 = np.array(K226_fit.fittedvalues);
aaa3 = np.array(JQ2J_fit.fittedvalues); aaa4 = np.array(K54D_fit.fittedvalues)


#Second Model
mod2 = 'FTSE_Y ~ EAFV + K226 + JQ2J + K54D + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 '
mod2_result = ols(mod2, data = FTSE_reg).fit()
bb0 = mod2_result.params.Intercept; bb1 = mod2_result.params.EAFV; bb2 = mod2_result.params.K226;
bb3 = mod2_result.params.JQ2J; bb4 = mod2_result.params.K54D

#Third Model
mod3 = 'FTSE_Y ~ EAFV + K226 + JQ2J + K54D + D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 + time'
mod3_result = ols(mod3, data = FTSE_reg).fit()
bbb0 = mod3_result.params.Intercept; bbb1 = mod3_result.params.EAFV; bbb2 = mod3_result.params.K226;
bbb3 = mod3_result.params.JQ2J; bbb4 = mod3_result.params.K54D

### Combine the fitted results of regression and forecasting results altogether

#Model 1
F = a1
for i in range(len(EAFV_train)):
    F[i] = b0 + a1[i]*b1 + a2[i]*b2 + a3[i]*b3 + a4[i]*b4

#Model 2
FF = aa1
for i in range(len(EAFV_train)):
    FF[i] = bb0 + aa1[i]*bb1 + aa2[i]*bb2 + aa3[i]*bb3 + aa4[i]*bb4

#Model 3
FFF = aaa1
for i in range(len(EAFV_train)):
    FFF[i] = bbb0 + aaa1[i]*bbb1 + aaa2[i]*bbb2 + aaa3[i]*bbb3 + aaa4[i]*bbb4


### The predicting results of 3 models on test set 
v1 = np.array(EAFV_Fcast); v2 = np.array(K226_Fcast);v3 = np.array(JQ2J_Fcast); v4 = np.array(K54D_Fcast)

vv1 = np.array(EAFV_Fcast); vv2 = np.array(K226_Fcast);
vv3 = np.array(JQ2J_Fcast); vv4 = np.array(K54D_Fcast)

vvv1 = np.array(EAFV_Fcast); vvv2 = np.array(K226_Fcast);
vvv3 = np.array(JQ2J_Fcast); vvv4 = np.array(K54D_Fcast)


#Model 1
E = v1
for i in range(len(EAFV_test)):
    E[i] = b0 + v1[i]*b1 + v2[i]*b2 + v3[i]*b3 + v4[i]*b4


#Model 2
EE = vv1
for i in range(len(EAFV_test)):
    EE[i] = bb0 + vv1[i]*bb1 + vv2[i]*bb2 + vv3[i]*bb3 + vv4[i]*bb4


#Model 3
EEE = vvv1
for i in range(len(EAFV_test)):
    EEE[i] = bbb0 + vvv1[i]*bbb1 + vvv2[i]*bbb2 + vvv3[i]*bbb3 + vvv4[i]*bbb4
    
    
# Append the fitted and predicting data for 3 models
K1 = np.append(F, E) #Model 1
K2 = np.append(FF, EE) #Model 2
K3 = np.append(FFF, EEE) #Model 3


#Real data
FTSE0 = read_excel('FTSEdata_31324878.xls', sheet_name='FTSE_reg', header=0, 
                     squeeze=True, dtype = float)
FTSE = FTSE0.FTSE
FTSE_train = FTSE[0:170]; FTSE_test = FTSE[170:239]


###############################

# Plotting the graphs 
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(K1, color='red', label='Forecast values')
line2, = plt.plot(FTSE, color='black', label='Original data')
#line3, = plt.plot(ci_low, color='blue', label='95% Confidence Interval Upper Bound')
#line4, = plt.plot(ci_low, color='orange', label='95% Confidence Interval Lower Bound')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.title('Model 1: FTSE Regression Forecast on Test Set')
plt.show()

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(K2, color='red', label='Forecast values')
line2, = plt.plot(FTSE, color='black', label='Original data')
#line3, = plt.plot(ci_low, color='blue', label='95% Confidence Interval Upper Bound')
#line4, = plt.plot(ci_low, color='orange', label='95% Confidence Interval Lower Bound')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.title('Model 2: FTSE Regression Forecast on Test Set')
plt.show()

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(K3, color='red', label='Forecast values')
line2, = plt.plot(FTSE, color='black', label='Original data')
#line3, = plt.plot(ci_low, color='blue', label='95% Confidence Interval Upper Bound')
#line4, = plt.plot(ci_low, color='orange', label='95% Confidence Interval Lower Bound')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.title('Model 3: FTSE Regression Forecast on Test Set')
plt.show()


