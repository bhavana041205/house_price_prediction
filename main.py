# 1. Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 2. Load the California Housing dataset

data = fetch_california_housing()


# 3. Create a DataFrame

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["Price"] = data.target


# 4. Understand the data

print("Number of rows and columns:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())


# 5. Check correlation

print("\nCorrelation with Price:")
print(df.corr()["Price"].sort_values(ascending=False))


# 6. Visualize the data

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="MedInc",
    y="Price"
)

plt.title("Income vs House Price")
plt.xlabel("Median Income")
plt.ylabel("House Price")

plt.show()


# 7. Separate features and target

X = df.drop("Price", axis=1)
Y = df["Price"]


# 8. Split the data

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 9. Create the Linear Regression model

model = LinearRegression()


# 10. Train the model

model.fit(X_train, Y_train)


# 11. Predict house prices

predicted_prices = model.predict(X_test)


# 12. Check model performance

mae = mean_absolute_error(
    Y_test,
    predicted_prices
)

r2 = r2_score(
    Y_test,
    predicted_prices
)

print("\nModel Results")
print("-------------")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)


# 13. Compare actual and predicted prices

plt.figure(figsize=(8, 5))

plt.scatter(
    Y_test,
    predicted_prices,
    alpha=0.5
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()
