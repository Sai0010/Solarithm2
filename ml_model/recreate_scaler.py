from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import os

# Create a new StandardScaler
scaler = StandardScaler()

# Try to fit it with sample data if the CSV exists
data_path = os.path.join(os.path.dirname(__file__), "data", "solar_data.csv")
if os.path.exists(data_path):
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    # Use the correct column names from the dataset
    features = data[['voltage', 'current', 'temp', 'lux']]
    # Fit the scaler with actual data
    scaler.fit(features)
    print(f"Scaler fitted with {len(features)} samples")
else:
    print("Data file not found, creating empty scaler")

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=4)  # Use protocol 4 for better compatibility
    
print("Scaler saved successfully to scaler.pkl")