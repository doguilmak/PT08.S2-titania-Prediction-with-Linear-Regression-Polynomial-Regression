"""
@author: doguilmak
Making prediction of PT08.S2(NMHC) Titania from Benzene concentration.

Attribute Information:

0 Date (DD/MM/YYYY)
1 Time (HH.MM.SS)
2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
7 True hourly averaged NOx concentration in ppb (reference analyzer)
8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
12 Temperature in Â°C
13 Relative Humidity (%)
14 AH Absolute Humidity

dataset: https://archive.ics.uci.edu/ml/datasets/Air+quality#

"""
#%%
# Importing Libraries

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import time

# %%
# Data Preprocessing

# Uploading Datas
start = time.time()
df = pd.read_csv('AirQualityUCI.csv')
df.info() # Looking for the missing values
# 2.2. Looking For Anomalies
print("\n", df.head())
print("\n", df.describe().T)
print("\n{} duplicated".format(df.duplicated().sum()))

# Creating correlation matrix heat map and examine relationship between datas
"""
Plot rectangular data as a color-encoded matrix.
https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""
plt.figure(figsize = (12,6))
sns.heatmap(df.corr(),annot = True)
#sns.pairplot(df)
plt.show()


# DataFrame Slice

x = df.iloc[0:500, 5:6]  # Benzene concentration in microg/m^3
y = df.iloc[0:500, 6:7]  # PT08.S2(NMHC) Titania

X = x.values
Y = y.values

#%%
# Linear Regression (Linear Model)

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

size = 3
plt.figure(figsize=(16, 8))
plt.scatter(X, Y, color='blue', s=size)
plt.plot(x, lin_reg.predict(X), color='orange')
plt.title('Linear Regression')
plt.xlabel('Benzene concentration in microg/m^3')
plt.ylabel('PT08.S2(NMHC) Titania')
sns.set_style("whitegrid")
plt.show()

# Mean Squared Error
from sklearn.metrics import mean_squared_error
print("\nLinear regression mean squared error: ", mean_squared_error(y, lin_reg.predict(X)))


# Polynomial Regression (Non-Linear Model/Quadratic)

# 2nd Order Polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, y)

# Mean Squared Error
print("\n2nd degree polynomial regression mean squared error: ", mean_squared_error(y, lin_reg2.predict(poly_reg2.fit_transform(X))))


# 4th Order Polynomial
poly_reg4 = PolynomialFeatures(degree = 4)
x_poly4 = poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4,y)

# Mean Squared Error
print("\n4th degree polynomial regression mean squared error: ", mean_squared_error(y, lin_reg4.predict(poly_reg4.fit_transform(X))))

# %%
# R² Value of the Regression Models

"""
What is R²?
The r2_score function computes the coefficient of determination, usually 
denoted as R².

It represents the proportion of variance (of y) that has been explained by 
the independent variables in the model. It provides an indication of goodness 
of fit and therefore a measure of how well unseen samples are likely to be 
predicted by the model, through the proportion of explained variance.

As such variance is dataset dependent, R² may not be meaningfully comparable 
across different datasets. Best possible score is 1.0 and it can be negative 
(because the model can be arbitrarily worse). A constant model that always 
predicts the expected value of y, disregarding the input features, would get a 
R² score of 0.0.

https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
"""

print('\n\nR² Values of the Regression:\n')
print('Linear R² value')
print(r2_score(Y, lin_reg.predict(X)))

print('\nPolynomial Regression R² value(degree=2)')
print(r2_score(Y, lin_reg2.predict(poly_reg2.fit_transform(X))))

print('\nPolynomial Regression R² value(degree=4)')
print(r2_score(Y, lin_reg4.predict(poly_reg4.fit_transform(X))))

#%%
# Prediction

# Titania prediction based on Benzene concentration in microg/m^3
print('\nPredictions:')

print("\nLinear Regression:\n")
print(lin_reg.predict([[20]]))
      
print("\nPolinomal Regression(degree=2):\n")
print(lin_reg2.predict(poly_reg2.fit_transform([[20]])))

print("\nPolinomal Regression(degree=4):\n")
print(lin_reg4.predict(poly_reg4.fit_transform([[20]])))

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
