import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.model_selection import cross_val_score


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

# Data preprocessing
X = data_encoded.drop('rings', axis=1)
y = data_encoded['rings']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Development
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Machine' : SVR()
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

#another way
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(f"Comparison for {name}:")
    print(comparison.head(10))  # Displaying the first 10 records for brevity
    print("\n")

# Plotting residuals
plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, label=name, alpha=0.7)

plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.legend()
plt.show()

# Model Evaluation: R^2 score
from sklearn.metrics import r2_score

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R^2 Score: {r2}")


#learning curve
   
plt.figure(figsize=(10, 6))

for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')
    train_scores = np.sqrt(-train_scores)
    test_scores = np.sqrt(-test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label=f'{name} Training score')
    plt.plot(train_sizes, test_scores_mean, label=f'{name} Validation score')

plt.xlabel('Training examples')
plt.ylabel('RMSE')
plt.title('Learning Curves')
plt.legend()
plt.grid()
plt.show()


# Evaluate models using cross-validation
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    cv_results[name] = cv_rmse

# Plotting boxplots for comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(cv_results))
plt.xlabel('Models')
plt.ylabel('Cross-Validated RMSE')
plt.title('Cross-Validated RMSE Comparison')
plt.show()
