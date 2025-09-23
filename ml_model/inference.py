import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class SolarPowerPredictor:
    """Predictor for solar power forecasting using trained LSTM model."""
    
    def __init__(self):
        """Initialize predictor with model and scaler."""
        self.base_dir = Path(__file__).parent
        self.model_path = self.base_dir / "trained_lstm.h5"
        self.scaler_path = self.base_dir / "scaler.pkl"
        self.config_path = self.base_dir / "train_config.json"
        
        # Check if model and scaler exist
        self.model_available = self._check_model_available()
        
        if self.model_available:
            logger.info("Loading LSTM model and scaler")
            self._load_artifacts()
        else:
            logger.warning("Model or scaler not found, will use baseline prediction")
    
    def _check_model_available(self):
        """Check if model and scaler files exist."""
        return (
            self.model_path.exists() and 
            self.scaler_path.exists() and 
            self.config_path.exists()
        )
    
    def _load_artifacts(self):
        """Load model, scaler, and configuration."""
        try:
            # Load model
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load scaler
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            
            # Load configuration
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            
            logger.info("Model artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            self.model_available = False
    
    def predict(self, recent_df, horizon=6):
        """
        Make predictions using LSTM model or fallback to baseline.
        
        Args:
            recent_df: DataFrame with recent sensor readings
            horizon: Number of future time steps to predict
            
        Returns:
            List of predicted power values
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
        logger.info(f"Making LSTM predictions for horizon={horizon}")
        
        # Extract features used during training
        features = self.config.get("features", ["voltage", "current", "temp", "lux"])
        
        # Ensure all required features are present
        for feature in features:
            if feature not in recent_df.columns:
                logger.warning(f"Feature '{feature}' not found in input data")
                return self._predict_baseline(recent_df, horizon)
        
        # Extract features from DataFrame
        feature_data = recent_df[features].values
        
        # Scale features
        scaled_features = self.scaler.transform(feature_data)
        
        # Prepare input shape for LSTM (samples, time_steps, features)
        window_size = self.config.get("window_size", 24)
        if len(scaled_features) < window_size:
            # Pad with zeros if not enough data
            pad_size = window_size - len(scaled_features)
            scaled_features = np.vstack([np.zeros((pad_size, len(features))), scaled_features])
        
        # Use the most recent window for prediction
        model_input = scaled_features[-window_size:].reshape(1, window_size, -1)
        
        # Make predictions for each step in the horizon
        predictions = []
        current_input = model_input.copy()
        
        for _ in range(horizon):
            # Predict next step
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update input for next prediction (rolling window)
            # This is a simplified approach - in a real system, we would need to
            # predict all features, not just power
            current_input = np.roll(current_input, -1, axis=1)
            
            # For simplicity, we're just shifting the window and not updating features
            # In a real system, we would need a more sophisticated approach
        
        logger.info(f"LSTM predictions completed: {predictions}")
        return predictions
    
    def _predict_baseline(self, recent_df, horizon):
        """
        Fallback baseline prediction using moving average.
        
        Simple moving average of power (voltage * current) over recent data.
        """
        logger.info(f"Using baseline prediction for horizon={horizon}")
        
        # Calculate power if not already in DataFrame
        if "power" in recent_df.columns:
            power = recent_df["power"].values
        elif "voltage" in recent_df.columns and "current" in recent_df.columns:
            power = recent_df["voltage"].values * recent_df["current"].values
        else:
            logger.warning("Cannot calculate power from input data")
            # Return zeros if we can't calculate power
            return [0.0] * horizon
        
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
        predictions = [avg_power * factor for factor in day_factors]
        
        logger.info(f"Baseline predictions completed: {predictions}")
        return predictions
    
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

# Function to be called from the API
def predict(recent_df, horizon=6):
    """
    Make power predictions using the trained model or baseline.
    
    Args:
        recent_df: DataFrame with recent sensor readings
        horizon: Number of future time steps to predict
        
    Returns:
        List of predicted power values
    """
    predictor = SolarPowerPredictor()
    return predictor.predict(recent_df, horizon)

if __name__ == "__main__":
    # Simple test
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=24, freq="H"),
        "voltage": np.random.uniform(11, 18, 24),
        "current": np.random.uniform(0.1, 4.5, 24),
        "temp": np.random.uniform(10, 30, 24),
        "lux": np.random.uniform(0, 1000, 24),
    }
    df = pd.DataFrame(data)
    df["power"] = df["voltage"] * df["current"]
    df.set_index("timestamp", inplace=True)
    
    # Make predictions
    predictions = predict(df, horizon=6)
    print(f"Predictions: {predictions}")