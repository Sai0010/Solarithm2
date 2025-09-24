import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class MLClient:
    """Client for ML model inference with fallback to baseline."""
    
    def __init__(self):
        """Initialize ML client, checking for model and scaler."""
        # Use absolute paths to ensure files are found regardless of working directory
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.model_path = base_dir / "ml_model" / "trained_lstm.keras"
        self.scaler_path = base_dir / "ml_model" / "scaler.pkl"
        self.config_path = base_dir / "ml_model" / "train_config.json"
        
        # Check if model and scaler exist
        self.model_available = self._check_model_available()
        
        if self.model_available:
            logger.info("ML model and scaler found, using LSTM for predictions")
            # Lazy import TensorFlow to avoid dependency if not needed
            import tensorflow as tf
            import pickle
            import json
            
            # Load model, scaler and config
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            logger.warning("ML model or scaler not found, will use baseline prediction")
    
    def _check_model_available(self):
        """Check if model and scaler files exist."""
        return self.model_path.exists() and self.scaler_path.exists() and self.config_path.exists()
    
    def predict(self, recent_df, horizon=6):
        """
        Make predictions using LSTM model or fallback to baseline.
        
        Args:
            recent_df: DataFrame with recent sensor readings
            horizon: Number of future time steps to predict
            
        Returns:
            List of predicted values
        """
        if not isinstance(recent_df, pd.DataFrame) or recent_df.empty:
            logger.error("Invalid input data for prediction")
            return []
        
        try:
            if self.model_available:
                return self._predict_with_model(recent_df, horizon)
            else:
                return self._predict_baseline(recent_df, horizon)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Fallback to baseline even if model is available but fails
            return self._predict_baseline(recent_df, horizon)
    
    def _predict_with_model(self, recent_df, horizon):
        """Make predictions using the LSTM model."""
        # Extract features used during training
        features = recent_df[['voltage_v', 'current_a', 'temp_c', 'lux']]
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Prepare input shape for LSTM (samples, time_steps, features)
        window_size = self.config.get('window_size', 24)
        if len(scaled_features) < window_size:
            # Pad with zeros if not enough data
            pad_size = window_size - len(scaled_features)
            scaled_features = np.vstack([np.zeros((pad_size, scaled_features.shape[1])), scaled_features])
        
        # Use the most recent window for prediction
        model_input = scaled_features[-window_size:].reshape(1, window_size, -1)
        
        # Make predictions for each step in the horizon
        predictions = []
        current_input = model_input.copy()
        
        for _ in range(horizon):
            # Predict next step
            next_pred = self.model.predict(current_input, verbose=0)[0]
            predictions.append(next_pred)
            
            # Update input for next prediction (rolling window)
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, :] = next_pred
        
        # Inverse transform to get actual values
        predictions = np.array(predictions).reshape(horizon, -1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Return power predictions (voltage * current)
        return predictions[:, 0] * predictions[:, 1]  # voltage * current
    
    def _predict_baseline(self, recent_df, horizon):
        """
        Fallback baseline prediction using moving average.
        
        Simple moving average of power (voltage * current) over recent data.
        """
        # Calculate power
        if 'power' in recent_df.columns:
            power = recent_df['power'].values
        else:
            power = recent_df['voltage_v'].values * recent_df['current_a'].values
        
        # If we have enough data, use moving average
        if len(power) >= 3:
            # Simple moving average of the last 3 values
            avg_power = np.mean(power[-3:])
        else:
            # Not enough data, use the last value or 0
            avg_power = power[-1] if len(power) > 0 else 0
        
        # Apply simple day curve model for solar (bell curve)
        hour_of_day = pd.Timestamp.now().hour
        day_factors = self._get_day_factors(hour_of_day, horizon)
        
        # Apply day factors to average power
        return [avg_power * factor for factor in day_factors]
    
    def _get_day_factors(self, current_hour, horizon):
        """
        Generate day factors based on time of day.
        
        Creates a simple bell curve peaking at noon.
        """
        # Simple bell curve with peak at noon (hour 12)
        day_curve = [max(0, 1 - abs(h - 12) / 12) for h in range(24)]
        
        # Get factors for the next 'horizon' hours
        factors = []
        for i in range(horizon):
            hour = (current_hour + i) % 24
            factors.append(day_curve[hour])
        
        return factors