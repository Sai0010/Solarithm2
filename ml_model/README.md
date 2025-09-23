# SolArithm ML Pipeline

This directory contains the machine learning pipeline for solar power forecasting.

## Data

Place your solar sensor data in the `data/` directory. The expected format is a CSV file with the following columns:
- `timestamp`: Date and time of the reading
- `voltage`: Voltage reading in volts
- `current`: Current reading in amps
- `temp`: Temperature in Celsius
- `lux`: Light intensity in lux
- `power`: Power in watts (voltage * current)
- `humidity`: Humidity percentage (optional)

## Training

To train the LSTM model, run:

```bash
python train_lstm.py
```

This will:
1. Load and preprocess data from `data/solar_data.csv`
2. Create sequences for LSTM training
3. Build and train a 2-layer LSTM model
4. Save the trained model to `trained_lstm.h5`
5. Save the scaler to `scaler.pkl`
6. Save the configuration to `train_config.json`

## Inference

To make predictions using the trained model, use the `inference.py` module:

```python
from ml_model.inference import predict

# Create a DataFrame with recent sensor readings
recent_df = pd.DataFrame({...})

# Make predictions for the next 6 hours
predictions = predict(recent_df, horizon=6)
```

If the trained model is not available, the system will automatically fall back to a baseline prediction using a moving average approach.

## Model Architecture

The LSTM model consists of:
- Input layer with shape (window_size, n_features)
- First LSTM layer with 64 units and dropout
- Second LSTM layer with 32 units and dropout
- Dense output layer with 1 unit

The model is trained with early stopping to prevent overfitting.