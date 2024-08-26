# PropertyPricePredictor-360
**PropertyPricePredictor-360** is a machine learning project that predicts property prices based on features such as size, number of bedrooms, and age of the property. This project demonstrates basic data preprocessing, model training, and evaluation using Python's `pandas` and `scikit-learn` libraries.

## Project Structure

- **`property_price_predictor.py`**: The main script containing the data preprocessing, model training, and prediction logic.
- **`requirements.txt`**: List of dependencies required for the project.

## Functionality

1. **Data Preparation**
   - **Description**: Loads the dataset and prepares the features (`Size`, `Bedrooms`, `Age`) and target variable (`Price`).

2. **Model Training**
   - **Description**: Splits the data into training and testing sets, trains a linear regression model, and evaluates its performance using Mean Squared Error (MSE).

3. **Prediction**
   - **Description**: Uses the trained model to predict the price of a new property based on its features.

## Code 
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
data = {
    'Size': [1500, 1600, 1700, 1800, 1900],
    'Bedrooms': [3, 3, 3, 4, 4],
    'Age': [10, 15, 20, 25, 30],
    'Price': [300000, 320000, 340000, 360000, 380000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Predict the price of a new property
new_house = pd.DataFrame([[2000, 4, 15]], columns=['Size', 'Bedrooms', 'Age'])
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")
