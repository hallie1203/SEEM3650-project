import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_and_clean_data(filepath):
    data = pd.read_excel(filepath, header=2)  # Assuming row 3 has the correct headers
    data.columns = ['Year', 'Male New Cases', 'Male Crude Rate', 'Male Age-Standardized Rate',
                    'Female New Cases', 'Female Crude Rate', 'Female Age-Standardized Rate']
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')  # Convert 'Year' to numeric
    data = data.dropna(subset=['Year'])  # Drop rows where 'Year' could not be converted
    return data


def prepare_data(data, rate_column):
    X = data['Year'].values.reshape(-1, 1)  # Year as independent variable
    y = data[rate_column].values  # Rate as dependent variable
    return X, y


def svm_regression_and_plot(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, y_pred, color='red', label='Predicted by SVM')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()

    print(f"MSE: {mean_squared_error(y, y_pred)}, R²: {r2_score(y, y_pred)}")


file_path = 'HKCaR-incidence-EN.xlsx'
data = load_and_clean_data(file_path)

# Plot for males
X_male, y_male = prepare_data(data, 'Male Age-Standardized Rate')
svm_regression_and_plot(X_male, y_male, 'SVM Regression: Male Age-Standardized Rate Over Time')

# Plot for females
X_female, y_female = prepare_data(data, 'Female Age-Standardized Rate')
svm_regression_and_plot(X_female, y_female, 'SVM Regression: Female Age-Standardized Rate Over Time')
