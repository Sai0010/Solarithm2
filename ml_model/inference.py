import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

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
            
            # Extract configuration parameters
            self.sequence_length = self.config.get("sequence_length", 24)
            self.features = self.config.get("features", ["voltage_v", "current_a", "temp_c", "humidity_pct", "lux"])
            self.target = self.config.get("target", "power_w")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
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
        if not self.model_available or recent_df.empty:
            logger.warning("Using baseline prediction due to missing model or data")
            return self._baseline_prediction(horizon)
        
        try:
            # Prepare input data
            X = self._prepare_input_data(recent_df)
            
            # Make prediction
            predictions = []
            current_input = X
            
            for _ in range(horizon):
                # Predict next step
                next_pred = self.model.predict(current_input, verbose=0)[0, -1]
                predictions.append(float(next_pred))
                
                # Update input for next prediction (rolling window)
                next_input = np.append(current_input[0, 1:, :], [[next_pred]], axis=0)
                current_input = np.array([next_input])
             
             # Inverse transform predictions
                if hasattr(self, 'scaler'):
                 # Create a dummy array with the right shape for inverse transform
                 dummy = np.zeros((len(predictions), len(self.features) + 1))
                 dummy[:, -1] = predictions  # Assuming target is the last column
                 dummy_inversed = self.scaler.inverse_transform(dummy)
                 predictions = dummy_inversed[:, -1].tolist()
             
                 return predictions
             
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self._baseline_prediction(horizon)
    
    def _prepare_input_data(self, df):
        """
        Prepare input data for LSTM model.
        
        Args:
            df: DataFrame with recent sensor readings
            
        Returns:
            Numpy array with shape (1, sequence_length, n_features)
        """
        # Ensure we have the required features
        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"Required feature {feature} not in input data")
        
        # Select only the required features
        df_features = df[self.features].copy()
        
        # Scale the features
        if hasattr(self, 'scaler'):
            df_features_scaled = pd.DataFrame(
                self.scaler.transform(df_features),
                columns=self.features
            )
        else:
            # Simple min-max scaling if no scaler available
            df_features_scaled = (df_features - df_features.min()) / (df_features.max() - df_features.min() + 1e-10)
        
        # Get the last sequence_length rows
        if len(df_features_scaled) >= self.sequence_length:
            recent_data = df_features_scaled.iloc[-self.sequence_length:].values
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.sequence_length - len(df_features_scaled), len(self.features)))
            recent_data = np.vstack([padding, df_features_scaled.values])
        
        # Reshape for LSTM input [samples, time steps, features]
        return np.array([recent_data])
    
    def _baseline_prediction(self, horizon):
        """
        Generate baseline prediction when model is not available.
        Uses a simple time-of-day based heuristic.
        
        Args:
            horizon: Number of future time steps to predict
            
        Returns:
            List of predicted power values
        """
        # Get current time
        now = datetime.now()
        predictions = []
        
        for i in range(horizon):
            # Predict for each hour in the horizon
            future_time = now + timedelta(hours=i)
            hour = future_time.hour
            
            # Simple heuristic based on time of day
            if 6 <= hour < 10:  # Morning ramp-up
                power = 200 + (hour - 6) * 100
            elif 10 <= hour < 16:  # Peak production
                power = 600 - abs(hour - 13) * 50
            elif 16 <= hour < 20:  # Evening ramp-down
                power = 400 - (hour - 16) * 100
            else:  # Night
                power = 0
                
            # Add some randomness
            power = max(0, power * (0.9 + 0.2 * np.random.random()))
            predictions.append(float(power))
            
        return predictions
    
    def forecast(self, start_time, horizon=24):
        """
        Generate power forecast for a specific time period.
        
        Args:
            start_time: Datetime for the start of the forecast
            horizon: Number of hours to forecast
            
        Returns:
            DataFrame with timestamp and predicted power
        """
        # Get recent data from database (would be implemented in a real system)
        # For now, we'll use the baseline prediction
        predictions = self._baseline_prediction(horizon)
        
        # Create timestamps for the forecast period
        timestamps = [start_time + timedelta(hours=i) for i in range(horizon)]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            'power_w': predictions
        })
        
        return forecast_df
    
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
        if "power_w" in recent_df.columns:
            power = recent_df["power_w"].values
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