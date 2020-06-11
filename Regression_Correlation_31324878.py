# -*- coding: utf-8 -*-



#Regression Preliminary
from pandas import read_excel
import matplotlib.pyplot as plt
import pandas as pd
FTSE = read_excel('FTSEdata_31324878.xls', sheet_name='FTSE_ori', header=0, 
                     squeeze=True)

#Plotting the scatter plots of each variable against the other one
pd.plotting.scatter_matrix(FTSE, figsize=(8, 8))
plt.show()

# Correlation matrix for all the variables, 2 by 2
CorrMatr = FTSE.corr()
print(CorrMatr)


