# -*- coding: utf-8 -*-



####Question2
#ARIMA on K54D

#Differencing and Model Selection(testing)
import statsmodels.api as sm  
from pandas import read_excel
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('fivethirtyeight')

K54D = read_excel('K54Ddata_31324878.xls', sheet_name = 'K54D', index_col = 0, header = 0, parse_dates = True, squeeze = True)
#Detrend: 1st Differencing
dat = K54D.values


#Automatic Test of Parameters 
import warnings
import itertools
p = q = range(0,4)
d = range(1,2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
###Caution:Running this code may take time by 10-15 minutes depending on the device.
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#print(len(pdq))
warnings.filterwarnings("ignore") # specify to ignore warning messages

collectAICs=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(dat,
                                            order=param,
                                            seasonal_order=param_seasonal,                                        
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


