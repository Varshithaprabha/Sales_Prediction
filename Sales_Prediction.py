# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# For this example, let's assume the CSV file 'sales_data.csv' is in the same directory.
# The dataset should have columns: Advertising Spend, Customer Segment, Platform, and Sales.
data = pd.read_csv('Advertising.csv')

# Step 1.1: Check the column names to make sure they are correct
print("Data columns:")
print(data.columns)

# Step 2: Preprocess the data
# Strip any extra spaces from the column names
data.columns = data.columns.str.strip()

# Check again if the columns are as expected
print("Cleaned Data columns:")
print(data.columns)

# Step 3: Ensure that the columns exist before applying one-hot encoding
if 'Customer Segment' in data.columns and 'Platform' in data.columns:
    data = pd.get_dummies(data, columns=['Customer Segment', 'Platform'], drop_first=True)
else:
    print("The columns 'Customer Segment' and 'Platform' are missing from the dataset.")
    # Optionally, print all available columns if the ones above are missing
    print("Available columns:", data.columns)

# Step 4: Define the feature matrix (X) and target vector (y)
X = data.drop('Sales', axis=1)  # Features (independent variables)
y = data['Sales']  # Target (dependent variable)

# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared
r2 = model.score(X_test, y_test)
print(f"R-squared: {r2}")

# Step 9: Visualize the Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# Step 10: Visualize Feature Importance (Coefficients)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coefficients.sort_values('Coefficient', ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title('Feature Importance (Coefficients)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()

# Optionally, print the model coefficients for each feature
print("Model coefficients:")
print(coefficients)
