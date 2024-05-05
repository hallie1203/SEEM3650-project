import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_excel("HKCaR-incidence-EN.xlsx",sheet_name='1')
# Select the relevant columns for the analysis (incidence rate)
X_male = data['M'].values.reshape(-1, 1)
y_male = data['Y'].values.reshape(-1, 1)
data2 = pd.read_excel("HKCaR-incidence-EN.xlsx",sheet_name='2')
X_female = data2['F'].values.reshape(-1, 1)
y_female = data2['Y'].values.reshape(-1, 1)


# Create separate instances of the LinearRegression model for each gender
model_male = LinearRegression()
model_female = LinearRegression()

# Fit the linear regression models
model_male.fit(X_male, y_male)
model_female.fit(X_female, y_female)

# Get the coefficients and intercept for each gender
coefficients_male = model_male.coef_
intercept_male = model_male.intercept_

coefficients_female = model_female.coef_
intercept_female = model_female.intercept_

# Print the coefficients and intercept for each gender
print('Male Coefficients:', coefficients_male)
print('Male Intercept:', intercept_male)

print('Female Coefficients:', coefficients_female)
print('Female Intercept:', intercept_female)