# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:54:28 2020

@author: mjz_inter
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

#read dataset
customers = pd.read_csv('Ecommerce Customers')

#Display basic info from dataset 
display(customers.head())
display(customers.describe())
customers.info()

'''
Exploring the dataset
'''
#compare the Time on Website and Yearly Amount Spent columns
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers, color='pink')
#compare the Time on App and Yearly Amount Spent columns
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers, color='pink')
#comparing Time on App and Length of Membership
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, color='pink', kind='hex')
# explore relationships across the entire dataset
sns.pairplot(customers, palette="husl")
#linear model plot of Yearly Amount Spent vs. Length of Membership
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data= customers, palette="husl")
plt.show()

'''
Training the model
'''
#divide dataset  into features (X) and labels (y)
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

#solit dataset into taring and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#traing the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

#printing model coefficients
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print('\n model coeffs: ', coeff_df)

'''
Testing the model
'''
pred = lm.predict(X_test)
#plot of the real test values versus the predicted values
plt.scatter(y_test, pred, c='pink', edgecolors='r', marker='.')
plt.xlabel('Y Test')
plt.ylabel('Predictions')
plt.grid()
plt.show()

'''
Model evaluation
'''
#Error calculations 
from sklearn import metrics
print('---------------\n')
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

#histogram of the residuals
sns.distplot((y_test-pred), bins= 50, color='pink')
plt.grid()

