import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanAbsolutePercentageError
import joblib  # For saving the scaler

# Step 1: Load and preprocess data
data = pd.read_csv(r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\data.csv')

# Handle missing values (if any)
data = data.fillna(data.mean())  # Impute with mean values

# Feature selection
X = data.drop(columns=['Main engine consumption (L/hr)'])
y = data['Main engine consumption (L/hr)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler to a file
scaler_path = r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at: {scaler_path}")

# Step 2: Define the model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Step 3: Compile and train the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', MeanAbsolutePercentageError()])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Save the trained model
model_path = r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\fuel_prediction_model.h5'
model.save(model_path)
print(f"Model saved at: {model_path}")

# Step 4: Evaluate the model
loss, mae, mape = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f} L/hr")
print(f"Test MAPE: {mape:.2f}%")

# Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Step 5: Get user input and make predictions
def predict_new_data():
    # Prompt user for input data
    print("Enter new data values for prediction:")
    feature_names = X.columns
    new_data = []
    for feature in feature_names:
        value = float(input(f"Enter {feature}: "))
        new_data.append(value)

    # Preprocess new data
    new_data = np.array(new_data).reshape(1, -1)
    new_data_scaled = scaler.transform(new_data)

    # Predict fuel consumption
    prediction = model.predict(new_data_scaled)
    print(f"Predicted Fuel Consumption: {prediction[0][0]:.2f} L/hr")

# Test prediction on new data
predict_new_data()