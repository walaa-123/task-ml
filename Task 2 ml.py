#!/usr/bin/env python
# coding: utf-8
2.
https://www.kaggle.com/datasets/budincsevity/szeged-weather/data
weatherHistory.csv

This is a regression dataset where:
The CSV file includes a hourly/daily summary for Szeged, Hungary area, between 2006 and 2016.
	Data available in the hourly response:
		time
		summary
		precipType
		temperature
		apparentTemperature
		humidity
		windSpeed
		windBearing
		visibility
		loudCover
		pressure
	Predict the tempreture using Multiple Regression and Polynomial regression.# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# In[1]:


# import libs : 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


# load data
df = pd.read_csv('weatherHistory.csv')
df.sample(10)     # print 10 random rows from the DataFrame
df.info()         # information of data like (null value , data type)

# describe data (min , max , mean values & IQR)
df.describe().T  
df.shape            # return number of row & col
df.shape[1]-1       # how many features in this data  &    -1 represent label of prediction


# In[3]:


# Create box plots for every variable 
plt.figure(figsize=(12, 8))
sns.set(style="darkgrid")
sns.boxplot(data=df, orient="h")
plt.title("Box Plot for Every Variable")
plt.show()
print(df.shape)


# In[4]:


# outlayers 
q1 = df['Wind Speed (km/h)'].quantile(0.25)
q3 = df['Wind Speed (km/h)'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['Wind Speed (km/h)'] >= lower_bound) & (df['Wind Speed (km/h)'] <= upper_bound)]


# In[5]:


# Create box plots for every variable after remove outlayers 
plt.figure(figsize=(12, 8))
sns.set(style="darkgrid")
sns.boxplot(data=df, orient="h")
plt.title("Box Plot for Every Variable")
plt.show()
print(df.shape)


# In[10]:


df['Precip Type'].unique()

df["Precip Type"] = df["Precip Type"].replace("rain", 1)
df["Precip Type"] = df["Precip Type"].replace("snow", 0)
df['Precip Type'].isnull().sum()      # 514
df.dropna(inplace=True)
df['Precip Type'].isnull().sum()       # 0    --> NaN values are dropped


# In[7]:


#creating a corr. heatmap to understand the pairwise relationships between variables 
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot =True, cmap="viridis", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# # Multiple Linear Regression
# * is a case of linear regression with two or more independent variables.
# *  the estimated regression function is 𝑓(𝑥₁, 𝑥₂) = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂ +......

# In[8]:


df.info()


# In[11]:


# define dependent & independent variables. (x,y)

#x= df.drop(["Apparent Temperature (C)" , "Humidity" ,"Visibility (km)"] , axis =1)
x =df.iloc[:,[2,5,8]]
y = df.iloc[:,3]


# In[12]:


# import train_test_split
from sklearn.model_selection import train_test_split
# split data into train & test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


# In[13]:


# import model
from sklearn.linear_model import LinearRegression
# creat an object 
model = LinearRegression()
#learn the model
model.fit(x_train,y_train)


# In[14]:


# get results: 
r_sq = model.score(x, y)      #obtain the value of 𝑅² using .score()
print(f"coefficient of determination: {r_sq}")
#the values of the estimators of regression coefficients with .intercept_ & .coef_
print(f"intercept: {model.intercept_}")          #.intercept_ holds the bias 𝑏₀
print(f"coefficients: {model.coef_}")            #.coef_ is an array containing 𝑏₁ & 𝑏₂,...


# In[15]:


# Predict response
y_pred = model.predict(x_test)
print(f"predicted response:\n{y_pred}")


# In[16]:


#visualization 
import seaborn as sns  
import matplotlib.pyplot as plt 

sns.pairplot(data = x_train, height = 2)  


# # Polynomial Regression
# * regression function 𝑓 can include nonlinear terms such as 𝑏₂𝑥₁², 𝑏₃𝑥₁³, or even 𝑏₄𝑥₁𝑥₂, 𝑏₅𝑥₁²𝑥₂.
# * The simplest example of polynomial regression has a single independent variable, and the estimated regression function is a polynomial of degree two: 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥 + 𝑏₂𝑥².
# 
# 

# In[43]:


# import Model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[44]:


# Features and the target variables
x = x =df.iloc[:,[2,5,8]].values
y = df.iloc[:, 3].values



# In[45]:


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)

#fit() we basically just declare what feature we want to transform
#transform() performs the actual transformation
x_poly = poly.fit_transform(x)
 


# In[46]:


# should be in the form [1, a, b, a^2, ab, b^2]
print(f'initial values {x[0]}\nMapped to {x_poly[0]}')


# In[41]:


#  fit the model
poly.fit(x_poly, y)

# we use linear regression as a base!!! ** sometimes misunderstood **
regression_model = LinearRegression()
regression_model.fit(x_poly, y)
y_pred = regression_model.predict(x_poly)
regression_model.coef_
print(f"coefficients: {regression_model.coef_}")   
mean_squared_error(y, y_pred, squared=False)
print(f"MSE: {mean_squared_error}") 


# In[52]:


plt.scatter(x[:, 0], y, color = 'red')
plt.plot(x[:,0], regression_model.predict(x_poly), color = 'blue') 
# plt.scatter(X[:,0], lin_reg_1.predict(poly_reg.fit_transform(X)), color = 'green')
plt.title('Multivariate Polynomial Regression')
plt.xlabel('first col of x')
plt.ylabel('temp')
plt.show()


# In[ ]:




