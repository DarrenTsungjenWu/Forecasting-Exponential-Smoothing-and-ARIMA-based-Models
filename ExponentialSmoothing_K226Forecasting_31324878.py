# -*- coding: utf-8 -*-

K226 = read_excel('K226data_31324878.xls', sheet_name = 'K226', index_col =0, header = 0, parse_dates = True, squeeze = True)

#Forecasting
K226_ffit = ExponentialSmoothing(K226, seasonal_periods = 12, trend = 'mul', seasonal = 'add').fit()
K226_FF = K226_ffit.forecast(12).rename('Model 4 Forecasting Future 12 Months')
error = K226 - K226_ffit.fittedvalues
mse = sum(error**2)/len(error)
ci_up = K226_FF + 1.96*mse
ci_low = K226_FF - 1.96*mse
ci_up.plot(color = 'green', label = '95% upper confidence interval', legend = True)
ci_low.plot(color = 'green', label = '95% lower confidence interval', legend = True)
K226.plot(color = 'black', label = 'Original Data', legend = True)
K226_FF.plot(color = 'green', legend = True)
pyplot.show()

#Residual Plot
resd = K226_ffit.resid
plot_acf(resd, lags = 50, title = 'ACF of K226 Residual')
plot_pacf(resd, lags = 50, title = 'PACF of K226 Residual')

