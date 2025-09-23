import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
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

class SolarLSTMTrainer:
    """LSTM model trainer for solar power forecasting."""
    
    def __init__(self, config=None):
        """Initialize trainer with configuration."""
        # Default configuration
        self.config = {
            "data_path": "data/solar_data.csv",
            "window_size": 24,  # 24 hours of data for prediction
            "train_split": 0.8,
            "resample_freq": "1H",  # 1 hour resampling
            "lstm_units": 64,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,  # Early stopping patience
            "features": ["voltage", "current", "temp", "lux"],
            "target": "power"
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Paths for saving model artifacts
        self.base_dir = Path(__file__).parent
        self.model_path = self.base_dir / "trained_lstm.h5"
        self.scaler_path = self.base_dir / "scaler.pkl"
        self.config_path = self.base_dir / "train_config.json"
        
        # Initialize attributes
        self.data = None
        self.scaler = None
        self.model = None
        self.history = None
    
    def load_data(self):
        """Load and preprocess the solar data."""
        logger.info(f"Loading data from {self.config['data_path']}")
        
        # Load data
        data_path = self.base_dir / self.config["data_path"]
        self.data = pd.read_csv(data_path)
        
        # Convert timestamp to datetime
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data.set_index("timestamp", inplace=True)
        
        # Resample if specified
        if self.config["resample_freq"]:
            logger.info(f"Resampling data to {self.config['resample_freq']} frequency")
            self.data = self.data.resample(self.config["resample_freq"]).mean()
        
        # Fill missing values
        self.data.fillna(method="ffill", inplace=True)
        self.data.fillna(method="bfill", inplace=True)
        
        logger.info(f"Data loaded and preprocessed: {len(self.data)} rows")
        return self.data
    
    def prepare_sequences(self):
        """Prepare sequences for LSTM training."""
        logger.info("Preparing sequences for LSTM training")
        
        # Extract features and target
        features = self.data[self.config["features"]].values
        target = self.data[self.config["target"]].values
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        window_size = self.config["window_size"]
        
        for i in range(len(scaled_features) - window_size):
            X.append(scaled_features[i:i+window_size])
            y.append(target[i+window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test
        train_size = int(len(X) * self.config["train_split"])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Sequences prepared: {len(X_train)} training, {len(X_test)} testing")
        return X_train, y_train, X_test, y_test
    
    def build_model(self):
        """Build the LSTM model."""
        logger.info("Building LSTM model")
        
        # Get input shape
        n_features = len(self.config["features"])
        window_size = self.config["window_size"]
        
        # Create model
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.config["lstm_units"],
            return_sequences=True,
            input_shape=(window_size, n_features)
        ))
        model.add(Dropout(self.config["dropout_rate"]))
        
        # Second LSTM layer
        model.add(LSTM(units=self.config["lstm_units"] // 2))
        model.add(Dropout(self.config["dropout_rate"]))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss="mse"
        )
        
        self.model = model
        logger.info(f"Model built: {model.summary()}")
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the LSTM model."""
        logger.info("Training LSTM model")
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config["patience"],
            restore_best_weights=True
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Model trained: test loss = {test_loss:.4f}")
        return self.history
    
    def save_artifacts(self):
        """Save model, scaler, and configuration."""
        logger.info("Saving model artifacts")
        
        # Save model
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        # Save scaler
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {self.scaler_path}")
        
        # Save configuration
        # Add input shape information to config
        save_config = self.config.copy()
        save_config["n_features"] = len(self.config["features"])
        
        with open(self.config_path, "w") as f:
            json.dump(save_config, f, indent=4)
        logger.info(f"Configuration saved to {self.config_path}")
    
    def plot_history(self):
        """Plot training history."""
        if self.history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.history["loss"], label="Training Loss")
            plt.plot(self.history.history["val_loss"], label="Validation Loss")
            plt.title("LSTM Training History")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_path = self.base_dir / "training_history.png"
            plt.savefig(plot_path)
            logger.info(f"Training history plot saved to {plot_path}")
            plt.close()
    
    def run_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting LSTM training pipeline")
        
        # Load and preprocess data
        self.load_data()
        
        # Prepare sequences
        X_train, y_train, X_test, y_test = self.prepare_sequences()
        
        # Build model
        self.build_model()
        
        # Train model
        self.train_model(X_train, y_train, X_test, y_test)
        
        # Plot history
        self.plot_history()
        
        # Save artifacts
        self.save_artifacts()
        
        logger.info("LSTM training pipeline completed")

if __name__ == "__main__":
    # Run the training pipeline
    trainer = SolarLSTMTrainer()
    trainer.run_pipeline()