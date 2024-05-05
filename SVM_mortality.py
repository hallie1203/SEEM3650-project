import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    data = pd.read_excel(filepath, header=2)
    data.columns = ['Year', 'Male New Cases', 'Male Crude Rate', 'Male Age-Standardized Rate',
                    'Female New Cases', 'Female Crude Rate', 'Female Age-Standardized Rate']
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data = data.dropna()
    return data

def prepare_data(data, rate_column):
    X = data[['Year']]
    y = data[rate_column]
    return X, y

def svm_regression_and_plot(X, y, title):
    # Scaling features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.scatter(X['Year'], y, color='blue', label='Actual Data')
    plt.plot(X['Year'], y_pred, color='red', label='Predicted by SVM')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"MSE: {mse:.2f}, R²: {r2:.2f}")

file_path = 'HKCaR-mortality-ENG.xlsx'
data = load_and_clean_data(file_path)

# Plot for males
X_male, y_male = prepare_data(data, 'Male Age-Standardized Rate')
svm_regression_and_plot(X_male, y_male, 'SVM Regression: Male Age-Standardized Mortality Rate Over Time')

# Plot for females
X_female, y_female = prepare_data(data, 'Female Age-Standardized Rate')
svm_regression_and_plot(X_female, y_female, 'SVM Regression: Female Age-Standardized Mortality Rate Over Time')