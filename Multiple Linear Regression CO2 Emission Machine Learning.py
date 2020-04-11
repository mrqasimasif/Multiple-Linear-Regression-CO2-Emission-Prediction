#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pwd


# In[7]:


#Reading Dataset
ds = pd.read_csv(r"E:\NAVTCCAI\MyWork\models\Multiple-Linear-Regression-Co2-Emission-Prediction\FuelConsumptionCo2.csv")

#Printing head data
ds.head()


# In[16]:


#Data filtering
cds = ds[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#Data head printing
cds.head()


# In[29]:


plt.scatter(cds.FUELCONSUMPTION_CITY,cds.CO2EMISSIONS,color='green')
plt.xlabel('Fuel Consumption CITY')
plt.ylabel('Co2Emissions')
#plt.show()


# In[56]:


#Test Train Split
msk = np.random.rand(len(ds)) < 0.75
train = cds[msk]
test = cds[~msk]

#print(train.head())
#print(test.head())


# In[60]:


#DIsplay Train Data
plt.scatter(train.FUELCONSUMPTION_CITY,train.CO2EMISSIONS,color='green')
plt.xlabel('Fuel Consumption City')
plt.ylabel('Co2 Emissions')
plt.show()


# In[61]:


#Display Test Data
plt.scatter(test.FUELCONSUMPTION_CITY,test.CO2EMISSIONS,color='green')
plt.xlabel('Fuel Consumption City')
plt.ylabel('Co2 Emissions')
plt.show()


# In[108]:


#Model Training 1
from sklearn import linear_model
reg = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])

#Feeding the Test Data
reg.fit(x_train,y_train)
print("Intercept ", reg.intercept_)
print("Coefficient ", reg.coef_)


# In[114]:


#Model 1 Prediction and Oridnary Least Square Error
y_hat = reg.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f" %np.mean((y_hat - y_test) ** 2))
print("Variance score: %.2f" %reg.score(x_test,y_test))


# In[123]:


plt.scatter(train.ENGINESIZE,train.CYLINDERS,train.FUELCONSUMPTION_COMB, color='blue')
plt.plot(x_train, reg.intercept_ + reg.coef_[0][1]*x_train[0][1:] + reg.coef_[0][2]*x_train[0][2:], '-r')


# In[93]:


#Model Training 2
from sklearn import linear_model
reg = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])

#Feeding the Data          
reg.fit(x_train,y_train)
print("Intercpet ",reg.intercept_) 
print("Coefficeints ",reg.coef_)
                       


# In[100]:


#Model 2 Prediction and Ordinary Least Square Error

y_hat = reg.predict(test[['ENGINESIZE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x_test = np.asanyarray(test[['ENGINESIZE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

#Printing the Error
print("Residual sum of squares: %.2f" %np.mean((y_hat - y_test) ** 2))
print("Variance score: %.2f" %reg.score(x_test,y_test))


# In[ ]:




