import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("C:\\Users\\USER\Downloads\\task1\\mynta_page(1-99).csv")

# Convert Price column
df['Price'] = pd.to_numeric(
    df['Price'].astype(str).str.extract('(\d+)')[0],
    errors='coerce'
)

# Convert Discount column
df['Discount'] = pd.to_numeric(
    df['Discount'].astype(str).str.extract('(\d+)')[0],
    errors='coerce'
)

# Remove missing values
df.dropna(inplace=True)

# Features
X = df[['Price', 'Discount']]

# Target
y = df['Selling Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Linear Regression Model
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score :", r2_score(y_test, y_pred))
print("Mean Squared Error :", mean_squared_error(y_test, y_pred))

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Values")
plt.show()