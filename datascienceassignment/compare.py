from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset from the CSV file
file_path = 'abalone.csv'
data = pd.read_csv(file_path)

# Convert categorical 'sex' column to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['sex'], drop_first=True)

# Separate features (X) and target variable (y)
X = data.drop('rings', axis=1)  # Features
y = data['rings']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor(n_estimators=100)),
    ('SVR', SVR(kernel='rbf'))
]

# Plotting actual vs predicted ages for each model in the same figure
plt.figure(figsize=(10, 8))
for name, model in models:
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    plt.scatter(y_test, y_pred_test, alpha=0.5, label=name)

plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Comparison of Predicted Ages by Different Models')
plt.legend()
plt.grid(True)
plt.show()
