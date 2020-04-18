#!/usr/bin/env python
# coding: utf-8
# # CHAPTER 3 선형회귀분석을 이용한 데이터 분석
# ## 3.1 단일선형회귀분석
# ### 3.1.5 단일선형회귀분석 실습 - Basic 1
# In[3]:

# scikit learn 패키지에서 linear_model이라는 모듈을 불러들임
from sklearn import linear_model
#import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
# matplotlib에는 사전에 정의한 많은 스타일이 있다.
# ggplot도 그 중 하나임
# R에서 사용하는 ggplot을 흉내내는 스타일을 사용하겠다고 선언
matplotlib.style.use('ggplot')

# In[4]:
# 2차원 배열을 만들어 'data'라는 변수에 할당
data = {'x' : [13, 19, 16, 14, 15, 14],
        'y' : [40, 83, 62, 48, 58, 43]}
# data라는 변수의 값을 data frame 형태로 변환
data = pd.DataFrame(data)
data
data.plot(kind="scatter",  # 산점도를 그리시오
          x='x',           # 가로축은 x라고 라벨을 붙임
          y='y',           # 세로축은 y라고 라벨을 붙임
          figsize=(5,5),   # 가로 5인치, 세로 5인치 크기의 박스를 설정
          color="blue")    # 산점도 상의 점 색상을 파랑색으로 지정
# linear_model 모듈이 포함하고 있는 Linearregression() 함수를 'linear_regression'이라고 하는 변수에 할당
linear_regression = linear_model.LinearRegression()
# Linearregression()의 fit()이라는 함수를 이용하여 선형회귀 모델 훈련 실행
# 이 때 독립변수는 x, 종속변수는 y
linear_regression.fit(X = pd.DataFrame(data["x"]), y = data["y"])
# 선형 회귀식의 세로축 절편 'linear_regression.intercept_'를 구하여 출력한다.
print('a value = ', linear_regression.intercept_)
# 선형 회귀식의 기울기 'linear_regression.coef_'를 구하여 출력한다.
print('b balue =', linear_regression.coef_)
# 위에서 만들어진 선형회귀 모델을 적용하여 선형회귀 값을 구해본다.
# 그 값을 prediction에 할당한다.
prediction = linear_regression.predict(X = pd.DataFrame(data["x"]))
# 실제 y값과 예측한 y값을 비교하여 잔차(residuals)를 구한다.
residuals = data["y"] - prediction
print(residuals)
# 변수의 갯수(6개), 잔차의 평균값, 잔차의 표준편차, 최소값, 25% 값, 50% 값, 75% 값, 최대값을 출력한다.
residuals.describe()
# 잔차를 제곱하여 전체를 합침. 결과값을 SSE라는 변수에 할당
SSE = (residuals**2).sum()
print("SSE = ", SSE)
# y값의 표준편차를 제곱한 것을 모두 합침. 그 결과값을 SST라는 변수에 할당
SST = ((data["y"]-data["y"].mean())**2).sum()
print("SST = ", SST)
# 결정계수 R을 구함
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)
data.plot(kind="scatter",x="x",y="y",figsize=(5,5),color="red")
# Plot regression line
plt.plot(data["x"],prediction,color="blue")
# sklearn.metrics라는 패키지로부터 mean_squared_error 모듈을 불러들임
from sklearn.metrics import mean_squared_error
# 결정계수 R값을 구함
print('score = ', linear_regression.score(X = pd.DataFrame(data["x"]), y = data["y"]))
# 실제값(data[y])과 회귀식 값(prediction)의 차이의 제곱을 구함
print('Mean_Squared_Error = ', mean_squared_error(prediction, data['y']))
# Mean squared error의 제곱근 값을 구함
print('RMSE = ', mean_squared_error(prediction, data['y'])**0.5)



# In[11]: 
# ### 3.1.6  단일선형회귀분석 실습 - Basic 2
from sklearn import datasets
boston_house_prices = datasets.load_boston()
print(boston_house_prices.keys())
print(boston_house_prices.data.shape)
print(boston_house_prices.feature_names)
#print(boston_house_prices.target)
data=boston_house_prices['data']
target=boston_house_prices['target']
# by Uchang

import pandas as pd 
df = pd.DataFrame(data=boston_house_prices['data'], columns = boston_house_prices['feature_names'])
#df.to_csv('boston.txt', sep = ',', index = False)
df.to_excel('boston_house_prices.xlsx', index=False)
print(boston_house_prices.DESCR)

data_frame = pd.DataFrame(boston_house_prices.data)
data_frame.tail()
data_frame.columns = boston_house_prices.feature_names
data_frame.tail()
data_frame['Price'] = boston_house_prices.target
data_frame.tail()
data_frame.plot(kind="scatter", x="RM", y="Price", figsize=(6,6),
                color="blue", xlim = (4,8), ylim = (10,45))
X = pd.DataFrame(data_frame["RM"])
print(X)

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(data_frame["RM"]), y = data_frame["Price"])
prediction = linear_regression.predict(X = pd.DataFrame(data_frame["RM"]))
print('a value = ', linear_regression.intercept_)
print('b balue =', linear_regression.coef_)
residuals = data_frame["Price"] - prediction
residuals.describe()
SSE = (residuals**2).sum()
SST = ((data_frame["Price"]-data_frame["Price"].mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared) 

data_frame.plot(kind="scatter",x="RM",y="Price",figsize=(6,6),
                color="blue", xlim = (4,8), ylim = (10,45))
# Plot regression line
plt.plot(data_frame["RM"],prediction,color="red")
print('score = ', linear_regression.score(X = pd.DataFrame(data_frame["RM"]), y = data_frame["Price"]))
print('Mean_Squared_Error = ', mean_squared_error(prediction, data_frame["Price"]))
print('RMSE = ', mean_squared_error(prediction, data_frame["Price"])**0.5)

