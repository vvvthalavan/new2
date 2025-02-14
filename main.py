import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data (or replace with real sensor data)
np.random.seed(42)
inlet_temp = np.random.uniform(15, 35, 100)  # Inlet temperature (Celsius)
room_temp = inlet_temp * 0.8 + np.random.normal(0, 1, 100)  # Room temperature with some noise

data = pd.DataFrame({'Inlet_Temperature': inlet_temp, 'Room_Temperature': room_temp})

data.to_csv('data/room_temp_data.csv', index=False)

# Load dataset
df = pd.read_csv('data/room_temp_data.csv')
X = df[['Inlet_Temperature']]
y = df['Room_Temperature']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot results
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Inlet Temperature (°C)')
plt.ylabel('Room Temperature (°C)')
plt.legend()
plt.show()
