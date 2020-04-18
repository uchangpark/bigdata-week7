#!/usr/bin/env python
# coding: utf-8
# ## 3.2 다중선형회귀분석
# ### 3.2.3 다중선형회귀분석 실습 - Basic 1
# In[1]:
from sklearn import linear_model
#import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

# In[3]:
data = {'x1' : [13, 18, 17, 20, 22, 21],
        'x2' : [9, 7, 17, 11, 8, 10],
        'y' : [20, 22, 30, 27, 35, 32]}
data = pd.DataFrame(data)
X = data[['x1', 'x2']]
y = data['y']
data

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(X), y = y)
prediction = linear_regression.predict(X = pd.DataFrame(X))
print('a value = ', linear_regression.intercept_)
print('b balue = ', linear_regression.coef_)

residuals = y-prediction
residuals.describe()

SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)

from sklearn.metrics import mean_squared_error
print('score = ', linear_regression.score(X = pd.DataFrame(X), y=y))
print('Mean_Squared_Error = ', mean_squared_error(prediction, y))
print('RMSE = ', mean_squared_error(prediction, y)**0.5)




# In[8]:
# ### 3.2.4  다중선형회귀분석 실습 - Basic 2

from sklearn import datasets
boston_house_prices = datasets.load_boston()
print(boston_house_prices.keys())
print(boston_house_prices.data.shape)
print(boston_house_prices.feature_names)

print(boston_house_prices.DESCR)

X = pd.DataFrame(boston_house_prices.data)
X.tail()

X.columns = boston_house_prices.feature_names
X.tail()

X['Price'] = boston_house_prices.target
y = X.pop('Price')
X.tail()

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(X), y = y)
prediction = linear_regression.predict(X = pd.DataFrame(X))
print('a value = ', linear_regression.intercept_)
print('b balue =', linear_regression.coef_)

residuals = y-prediction
residuals.describe()

SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)

print('score = ', linear_regression.score(X = pd.DataFrame(X), y = y)) #score=R_squared
print('Mean_Squared_Error = ', mean_squared_error(prediction, y))
print('RMSE = ', mean_squared_error(prediction, y)**0.5)


