import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



# Load the dataset
data = pd.read_csv('abalone.csv')

# Explore Data Features and Distributions
print(data)  # Display all records in the dataset
print(data.describe())  # Descriptive statistics

# Pairplot - visualize distributions
sns.pairplot(data)
plt.show()
# Investigate Correlations
# One-hot encode 'Sex' column
data_encoded = pd.get_dummies(data, columns=[data.columns[0]], drop_first=True)

correlation_matrix = data_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

correlation_with_target = correlation_matrix['rings'].sort_values(ascending=False)
print(correlation_with_target)
# Model Development
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}


# Train models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
# Model Evaluation
for name, model in models.items():
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    print(f"{name} RMSE on train dataset: {rmse_train}")
    print(f"{name} RMSE on test dataset: {rmse_test}")
#compare results
plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    plt.scatter(y_test, y_pred, label=name, alpha=0.5)

plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Model Performance: Actual vs Predicted values')
plt.legend()
plt.show()
