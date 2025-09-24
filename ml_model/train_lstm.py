import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SolarLSTMTrainer:
    """Trainer for LSTM model for solar power forecasting."""
    
    def __init__(self, data_path=None, config=None):
        """Initialize trainer with data path and configuration."""
        self.base_dir = Path(__file__).parent
        self.data_path = data_path or self.base_dir / "data" / "solar_data.csv"
        
        # Default configuration
        self.config = {
            "sequence_length": 24,  # Hours of data to use for prediction
            "forecast_horizon": 6,  # Hours to predict ahead
            "test_size": 0.2,       # Fraction of data for testing
            "validation_size": 0.2, # Fraction of training data for validation
            "batch_size": 32,       # Mini-batch size
            "epochs": 50,           # Maximum epochs
            "patience": 10,         # Early stopping patience
            "learning_rate": 0.001, # Initial learning rate
            "dropout_rate": 0.2,    # Dropout rate
            "lstm_units": [64, 32], # LSTM layer units
            "features": ["voltage", "current", "temp", "humidity", "lux"],
            "target": "power"
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
            
        # Output paths
        self.model_path = self.base_dir / "trained_lstm.keras"
        self.scaler_path = self.base_dir / "scaler.pkl"
        self.config_path = self.base_dir / "train_config.json"
        
        # Initialize attributes
        self.data = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        """Load and preprocess the data."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Check if timestamp column exists and convert to datetime
            if 'ts' in self.data.columns:
                self.data['ts'] = pd.to_datetime(self.data['ts'])
                self.data = self.data.sort_values('ts')
            
            # Check for missing values
            missing = self.data.isnull().sum()
            if missing.sum() > 0:
                logger.warning(f"Missing values detected: {missing[missing > 0]}")
                logger.info("Filling missing values with forward fill then backward fill")
                self.data = self.data.fillna(method='ffill').fillna(method='bfill')
                
            logger.info(f"Data loaded successfully with shape {self.data.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def prepare_sequences(self):
        """Prepare sequences for LSTM training."""
        features = self.config["features"]
        target = self.config["target"]
        seq_length = self.config["sequence_length"]
        
        # Check if all features and target exist in data
        missing_cols = [col for col in features + [target] if col not in self.data.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            return None, None, None, None, None, None
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.data[features])
        scaled_df = pd.DataFrame(scaled_features, columns=features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(self.data) - seq_length):
            X.append(scaled_df.iloc[i:i+seq_length].values)
            y.append(self.data[target].iloc[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train, validation, and test sets
        test_size = int(len(X) * self.config["test_size"])
        train_val_X, test_X = X[:-test_size], X[-test_size:]
        train_val_y, test_y = y[:-test_size], y[-test_size:]
        
        val_size = int(len(train_val_X) * self.config["validation_size"])
        train_X, val_X = train_val_X[:-val_size], train_val_X[-val_size:]
        train_y, val_y = train_val_y[:-val_size], train_val_y[-val_size:]
        
        logger.info(f"Training set: {train_X.shape}, Validation set: {val_X.shape}, Test set: {test_X.shape}")
        return train_X, train_y, val_X, val_y, test_X, test_y
    
    def build_model(self, input_shape):
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # Add LSTM layers with specified units
        lstm_units = self.config["lstm_units"]
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # Return sequences for all but last LSTM layer
            if i == 0:
                model.add(LSTM(units, activation='relu', return_sequences=return_sequences, 
                              input_shape=input_shape))
            else:
                model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
            
            # Add dropout after each LSTM layer
            model.add(Dropout(self.config["dropout_rate"]))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model with Adam optimizer and learning rate
        optimizer = Adam(learning_rate=self.config["learning_rate"])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"Model built with {len(lstm_units)} LSTM layers")
        model.summary(print_fn=logger.info)
        return model
    
    def train(self):
        """Train the LSTM model with mini-batch processing."""
        if self.data is None and not self.load_data():
            logger.error("Failed to load data for training")
            return False
        
        # Prepare sequences
        train_data = self.prepare_sequences()
        if train_data is None:
            logger.error("Failed to prepare sequences for training")
            return False
        
        train_X, train_y, val_X, val_y, test_X, test_y = train_data
        
        # Build model
        input_shape = (train_X.shape[1], train_X.shape[2])
        self.model = self.build_model(input_shape)
        
        # Setup callbacks for training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["patience"],
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when training plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Save best model during training
            ModelCheckpoint(
                filepath=str(self.model_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with mini-batch processing
        logger.info(f"Starting model training with batch size {self.config['batch_size']}")
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=2
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Model training completed in {training_time}")
        
        # Evaluate on test set
        test_loss, test_mae = self.model.evaluate(test_X, test_y, verbose=0)
        logger.info(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
        
        # Save artifacts
        self.save_artifacts()
        
        return True
    
    def save_artifacts(self):
        """Save model, scaler, and configuration."""
        try:
            # Save model
            if self.model:
                self.model.save(self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            
            # Save scaler
            if self.scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"Scaler saved to {self.scaler_path}")
            
            # Save configuration
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")
            return False
    
    def plot_history(self, save_path=None):
        """Plot training history."""
        if not self.history:
            logger.warning("No training history available to plot")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
            
    def predict(self, input_data):
        """Make predictions using the trained model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
            
        if self.scaler is None:
            logger.error("Scaler not available")
            return None
            
        # Ensure input data has correct shape
        if len(input_data.shape) == 2:  # Single sample
            input_data = np.expand_dims(input_data, axis=0)
            
        # Make prediction
        predictions = self.model.predict(input_data)
        return predictions.flatten()
    
    def run(self):
        """Run the full training pipeline."""
        logger.info("Starting LSTM model training pipeline")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Aborting training.")
            return False
            
        # Train model
        if not self.train():
            logger.error("Failed to train model. Aborting.")
            return False
            
        # Plot and save training history
        history_plot_path = self.base_dir / "training_history.png"
        self.plot_history(save_path=history_plot_path)
        
        logger.info("Training pipeline completed successfully")
        return True
        
    def run(self):
        """Run the full training pipeline."""
        logger.info("Starting LSTM model training pipeline")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Aborting training.")
            return False
            
        # Train model
        if not self.train():
            logger.error("Failed to train model. Aborting.")
            return False
            
        # Plot and save training history
        history_plot_path = self.base_dir / "training_history.png"
        self.plot_history(save_path=history_plot_path)
        
        logger.info("Training pipeline completed successfully")
        return True


if __name__ == "__main__":
    # Create and run trainer
    trainer = SolarLSTMTrainer()
    trainer.run()
    
    def load_data(self):
        """Load and preprocess the data."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Check if timestamp column exists and convert to datetime
            if 'ts' in self.data.columns:
                self.data['ts'] = pd.to_datetime(self.data['ts'])
                self.data = self.data.sort_values('ts')
            
            # Check for missing values
            missing = self.data.isnull().sum()
            if missing.sum() > 0:
                logger.warning(f"Missing values detected: {missing[missing > 0]}")
                logger.info("Filling missing values with forward fill then backward fill")
                self.data = self.data.fillna(method='ffill').fillna(method='bfill')
                
            logger.info(f"Data loaded successfully with shape {self.data.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def prepare_sequences(self):
        """Prepare sequences for LSTM training."""
        features = self.config["features"]
        target = self.config["target"]
        seq_length = self.config["sequence_length"]
        
        # Check if all features and target exist in data
        missing_cols = [col for col in features + [target] if col not in self.data.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            return None, None, None, None, None, None
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.data[features])
        scaled_df = pd.DataFrame(scaled_features, columns=features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(self.data) - seq_length):
            X.append(scaled_df.iloc[i:i+seq_length].values)
            y.append(self.data[target].iloc[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train, validation, and test sets
        test_size = int(len(X) * self.config["test_size"])
        train_val_X, test_X = X[:-test_size], X[-test_size:]
        train_val_y, test_y = y[:-test_size], y[-test_size:]
        
        val_size = int(len(train_val_X) * self.config["validation_size"])
        train_X, val_X = train_val_X[:-val_size], train_val_X[-val_size:]
        train_y, val_y = train_val_y[:-val_size], train_val_y[-val_size:]
        
        logger.info(f"Training set: {train_X.shape}, Validation set: {val_X.shape}, Test set: {test_X.shape}")
        return train_X, train_y, val_X, val_y, test_X, test_y
    
    def build_model(self, input_shape):
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # Add LSTM layers with specified units
        lstm_units = self.config["lstm_units"]
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # Return sequences for all but last LSTM layer
            if i == 0:
                model.add(LSTM(units, activation='relu', return_sequences=return_sequences, 
                              input_shape=input_shape))
            else:
                model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
            
            # Add dropout after each LSTM layer
            model.add(Dropout(self.config["dropout_rate"]))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model with Adam optimizer and learning rate
        optimizer = Adam(learning_rate=self.config["learning_rate"])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"Model built with {len(lstm_units)} LSTM layers")
        model.summary(print_fn=logger.info)
        return model
    
    def train_model(self, train_X, train_y, val_X, val_y):
        """Train the LSTM model."""
        logger.info("Training LSTM model")
        
        # Setup callbacks for training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["patience"],
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when training plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Save best model during training
            ModelCheckpoint(
                filepath=str(self.model_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with mini-batch processing
        logger.info(f"Starting model training with batch size {self.config['batch_size']}")
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=2
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Model training completed in {training_time}")
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
        if not self.load_data():
            logger.error("Failed to load data. Aborting training.")
            return False
        
        # Train model
        if not self.train():
            logger.error("Failed to train model. Aborting.")
            return False
        
        # Plot history
        self.plot_history()
        
        # Save artifacts
        self.save_artifacts()
        
        logger.info("LSTM training pipeline completed")
        return True

if __name__ == "__main__":
    # Run the training pipeline
    trainer = SolarLSTMTrainer()
    trainer.run()