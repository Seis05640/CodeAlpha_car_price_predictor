# car_price_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('car data.csv')

# 2. Drop text columns that don’t add value
if 'Car_Name' in df.columns:
    df = df.drop(['Car_Name'], axis=1)

# 3. Check for missing values
print(df.isnull().sum())

# 4. Convert categorical columns to dummy variables
df_encoded = pd.get_dummies(df, drop_first=True)

print(df_encoded.head())

# 5. Define features (X) and target (y)
X = df_encoded.drop('Selling_Price', axis=1)
y = df_encoded['Selling_Price']

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predict on test data
# 1. Numerical inputs
Year = int(input("Enter year of car (e.g., 2015): "))
Present_Price = float(input("Enter present ex-showroom price (in lakhs): "))
Kms_Driven = int(input("Enter kms driven: "))
Owner = int(input("Enter number of previous owners: "))

# 2. Categorical inputs
Fuel_Type = input("Enter Fuel Type (Petrol/Diesel): ")
Seller_Type = input("Enter Seller Type (Dealer/Individual): ")
Transmission = input("Enter Transmission Type (Manual/Automatic): ")


# Dummy encoding: Must match the order used in training data
Fuel_Type_Diesel = 0
Fuel_Type_Petrol = 0
if Fuel_Type.lower() == 'petrol':
    Fuel_Type_Petrol = 1
elif Fuel_Type.lower() == 'diesel':
    Fuel_Type_Diesel = 1
# Else CNG = 0 for both dummy columns

Seller_Type_Individual = 1 if Seller_Type.lower() == 'individual' else 0
Transmission_Manual = 1 if Transmission.lower() == 'manual' else 0

# Create final input array
input_data = np.array([Year, Present_Price, Kms_Driven, Owner,
                       Fuel_Type_Diesel, Fuel_Type_Petrol,
                       Seller_Type_Individual, Transmission_Manual]).reshape(1, -1)

# Predict
predicted_price = model.predict(input_data)
print(f"\nEstimated Selling Price: ₹ {predicted_price[0]:.2f} lakhs")

# 9. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 10. Visualize Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()
