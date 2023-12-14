from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# Initialize the Support Vector Regression (SVR) model
model = SVR(kernel='rbf')  # Radial Basis Function (RBF) kernel; other kernels can be used

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Create DataFrames to compare actual vs predicted ages
results_train = pd.DataFrame({'Actual Age': y_train, 'Predicted Age': y_pred_train})
results_test = pd.DataFrame({'Actual Age': y_test, 'Predicted Age': y_pred_test})

# Display the first few rows of the comparison between actual and predicted ages
print("Training set - Actual vs Predicted Age:")
print(results_train.head())

print("\nTest set - Actual vs Predicted Age:")
print(results_test.head())

# Evaluate model performance
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

print("Root mean square error on train dataset:", rmse_train)
print("Root mean square error on test dataset:", rmse_test)

# Analysis and further steps to improve the model
# Investigate feature importance, consider feature engineering, or try other models for better performance

# Descriptive statistics - Correlation matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualization of data distribution and relationships
sns.pairplot(data)
plt.show()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#%%
