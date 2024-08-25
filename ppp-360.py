import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Size': [1500, 1600, 1700, 1800, 1900],
    'Bedrooms': [3, 3, 3, 4, 4],
    'Age': [10, 15, 20, 25, 30],
    'Price': [300000, 320000, 340000, 360000, 380000]
}

df = pd.DataFrame(data)
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Convert new_house to DataFrame
new_house = pd.DataFrame([[2000, 4, 15]], columns=['Size', 'Bedrooms', 'Age'])
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")
