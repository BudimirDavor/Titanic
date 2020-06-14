# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:55:31 2020
@author: Davor
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('titanic.csv')

#---Read in Your Data---#

print(df.head())
print(df.tail())

#---Summarize the Data, describe---#


pd.options.display.max_columns = 6
print(df.describe())

#---Selecting a Single Column---#

col = df['Fare']
print(col)

#---Selecting Multiple Columns---#

small_df = df[['Age', 'Sex', 'Survived']]
print(small_df.head())

#---Creating a Column---#

print(df['Sex'] == 'male')

#---Converting from a Pandas Series to a Numpy Array---#

print(df['Fare'].values)

#---Converting from a Pandas DataFrame to a Numpy Array---#

print(df[['Pclass', 'Fare', 'Age']].values)

#---Numpy Shape Attribute---#

arr = df[['Pclass', 'Fare', 'Age']].values
print(arr.shape)

#---Select from a Numpy Array---#

arr = df[['Pclass', 'Fare', 'Age']].values
print(arr)
print(arr[0, 1])
print(arr[0])
print(arr[:,2])

#---Masking---#

# take first 10 values for simplicity
arr = df[['Pclass', 'Fare', 'Age']].values[:10]
mask = arr[:, 2] < 18
print(arr[mask])
print(arr[arr[:, 2] < 18])

#---Summing and Counting---#

arr = df[['Pclass', 'Fare', 'Age']].values
mask = arr[:, 2] < 18
print(mask.sum())
print((arr[:, 2] < 18).sum())

#---Scatter Plot---#

plt.scatter(df['Age'], df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
plt.plot([0, 80], [85, 5])

#---Prep Data with Pandas---#

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
print(X)
print(y)

#---Build a Logistic Regression Model with Sklearn---#

X = df[['Fare', 'Age']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
print(model.coef_, model.intercept_)

#---Make Predictions with the Model---#

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))
print(model.predict(X[:5]))
print(y[:5])

#---Score the Model---#

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
print((y == y_pred).sum())














