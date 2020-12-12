# Forecasting-Exponential-Smoothing-and-ARIMA-based-Models

This is a Forecasting task to model some economic-and-stock-based data (details see .xls files attached) applying Exponential Smoothing (ES) and ARIMA models.

In this task, preliminary analysis, model selection/measurement and out-sample forecasting are implemented.

In this repository, there are 19 python files, 5 datasets and 1 techincal reports included.

The task has 3 seperate tasks for comparing and contrasting 3 different forecasting models on indicator prediction. The results are shown in pdf file "Technical Report".

Specifically, in first part, ES models are trained over 4 datasets: EAFV, JQ2J, K226 and K54D.
We reveal data structure using visulisation method and autocorrelation.
Subsequently we train ES model according to combinations of trend and seasonal components and select appropriate one with best model predictive performance over test set.

In second part, we train ARIMA model based on FTSE index (see FTSEdata_31324878.xls)and K54D respectively throughout detailed model selection visually and automatically. 
And the model performance is also compared to ES models.

In third part, an ensemble model blended with Linear Regression and ARIMA model is used so as to forecast FTSE based on all other indicator features.

For all details, please read TechnicalReport_31324878.pdf. The python file names are named by model_data_method, so that you can refer it to obtain corresponding results presented in each section of the report.


Last updated: 2020/06/12
