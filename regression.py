# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "cleaned.csv"
df = pd.read_csv(file_path)

# Data overview
print(df.info(), "\n")
print(df.head(), "\n")

# Handle missing values by replacing them with the mean of the respective column
df = df.fillna(df.mean())

# Drop all non-numeric features
df = df.select_dtypes(include=[np.number])

# Select features and target
features = []
target = 'Price'

if not all(col in df.columns for col in features + [target]):
    raise ValueError("Missing required columns in the dataset!")

X = df[features].values
y = df[target].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias term
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Initialize parameters
theta = np.zeros((X_train_scaled.shape[1], 1))

# Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = X.dot(theta)
        gradient = (1/m) * X.T.dot(predictions - y)
        theta -= alpha * gradient
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return theta, cost_history

# Train model
alpha, iterations = 0.01, 1000
theta, cost_history = gradient_descent(X_train_scaled, y_train, theta, alpha, iterations)

# Prediction function
def predict(X, theta):
    return X.dot(theta)

# Make predictions
y_pred = predict(X_test_scaled, theta)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Plot cost function convergence
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history)), cost_history, color='blue')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.show()

# Visualization: Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test.flatten(), y=y_pred.flatten(), alpha=0.7)
plt.xlabel("Actual Prices (NPR)")
plt.ylabel("Predicted Prices (NPR)")
plt.title("Actual vs Predicted House Prices")
plt.show()
