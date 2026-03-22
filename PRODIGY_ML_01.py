# Import libraries
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("train.csv")

# Select important features
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Remove missing values
data = data.dropna()

# Features (X) and Target (y)
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== Model Performance =====")
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# Display learned coefficients
print("\n===== Model Equation =====")
print("Price = b0 + b1*(SqFt) + b2*(Bedrooms) + b3*(Bathrooms)")

print("b0 (Intercept):", round(model.intercept_, 2))
print("b1 (SqFt):", round(model.coef_[0], 2))
print("b2 (Bedrooms):", round(model.coef_[1], 2))
print("b3 (Bathrooms):", round(model.coef_[2], 2))

# Visualization (Actual vs Predicted)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# User Input Prediction
print("\n===== Predict House Price =====")

try:
    sqft = float(input("Enter square footage: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))

    if sqft <= 0 or bedrooms < 0 or bathrooms < 0:
        print("❌ Invalid input! Values must be positive.")
    else:
        prediction = model.predict([[sqft, bedrooms, bathrooms]])
        print("\n✅ Estimated House Price: $", round(prediction[0], 2))

except:
    print("❌ Please enter valid numeric values!")