import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data from the JSON file
data = pd.read_excel('HKCaR-mortality-EN.xlsx', sheet_name='1')
data1 = pd.read_excel('HKCaR-mortality-EN.xlsx', sheet_name='2')
# Filter the data for male and female


# Prepare the data for training
X_male = data[['Y']].values
y_male = data['M'].values
X_female = data1[['Y']].values
y_female = data1['F'].values

# Split the data into training and testing sets
X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.2, random_state=42)
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female, test_size=0.2, random_state=42)

# Create and train the Linear Regression models
model_lr_male = LinearRegression()
model_lr_male.fit(X_train_male, y_train_male)
model_lr_female = LinearRegression()
model_lr_female.fit(X_train_female, y_train_female)

# Create and train the Gradient Boosting models
model_gb_male = GradientBoostingRegressor(random_state=42)
model_gb_male.fit(X_train_male, y_train_male)
model_gb_female = GradientBoostingRegressor(random_state=42)
model_gb_female.fit(X_train_female, y_train_female)

# Make predictions on the test sets
y_pred_lr_male = model_lr_male.predict(X_test_male)
y_pred_lr_female = model_lr_female.predict(X_test_female)
y_pred_gb_male = model_gb_male.predict(X_test_male)
y_pred_gb_female = model_gb_female.predict(X_test_female)

# Calculate the mean squared error for Linear Regression
mse_lr_male = mean_squared_error(y_test_male, y_pred_lr_male)
mse_lr_female = mean_squared_error(y_test_female, y_pred_lr_female)

# Calculate the mean squared error for Gradient Boosting
mse_gb_male = mean_squared_error(y_test_male, y_pred_gb_male)
mse_gb_female = mean_squared_error(y_test_female, y_pred_gb_female)

print("Linear Regression MSE for Male Mortality:", mse_lr_male)
print("Linear Regression MSE for Female Mortality:", mse_lr_female)
print("Gradient Boosting MSE for Male Mortality:", mse_gb_male)
print("Gradient Boosting MSE for Female Mortality:", mse_gb_female)
