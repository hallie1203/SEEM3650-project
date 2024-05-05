import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_excel('incidence.xlsx', sheet_name='1')
data1 = pd.read_excel('incidence.xlsx', sheet_name='2')

X_male = data[['Y']].values
y_male = data['M'].values
X_female = data1[['Y']].values
y_female = data1['F'].values

X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.2, random_state=42)
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_male_poly = poly.fit_transform(X_train_male)
X_test_male_poly = poly.transform(X_test_male)
X_train_female_poly = poly.fit_transform(X_train_female)
X_test_female_poly = poly.transform(X_test_female)

model_ridge_male = Ridge(alpha=1.0)
model_ridge_male.fit(X_train_male_poly, y_train_male)
model_ridge_female = Ridge(alpha=1.0)
model_ridge_female.fit(X_train_female_poly, y_train_female)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
}
gb_grid_search_male = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5)
gb_grid_search_female = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5)

gb_grid_search_male.fit(X_train_male, y_train_male)
gb_grid_search_female.fit(X_train_female, y_train_female)

best_gb_model_male = gb_grid_search_male.best_estimator_
best_gb_model_female = gb_grid_search_female.best_estimator_

y_pred_ridge_male = model_ridge_male.predict(X_test_male_poly)
y_pred_ridge_female = model_ridge_female.predict(X_test_female_poly)
y_pred_gb_male = best_gb_model_male.predict(X_test_male)
y_pred_gb_female = best_gb_model_female.predict(X_test_female)

mse_ridge_male = mean_squared_error(y_test_male, y_pred_ridge_male)
mse_ridge_female = mean_squared_error(y_test_female, y_pred_ridge_female)

mse_gb_male = mean_squared_error(y_test_male, y_pred_gb_male)
mse_gb_female = mean_squared_error(y_test_female, y_pred_gb_female)

print("Ridge Regression MSE for Male Incidence:", mse_ridge_male)
print("Ridge Regression MSE for Female Incidence:", mse_ridge_female)
print("Gradient Boosting MSE for Male Incidence:", mse_gb_male)
print("Gradient Boosting MSE for Female Incidence:", mse_gb_female)