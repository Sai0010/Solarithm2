# SolArithm Project Makefile

# Directories
APP_DIR = app
ML_DIR = ml_model
SCRIPTS_DIR = scripts
DATA_DIR = $(ML_DIR)/data
CONFIG_DIR = config
VENV_DIR = venv

# Python commands
PYTHON = python
PIP = pip

# Targets
.PHONY: all setup venv install generate-data train-model validate clean

all: setup generate-data train-model validate

# Setup environment and directories
setup: venv install init-dirs

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install:
	$(VENV_DIR)\Scripts\pip install -r requirements.txt

init-dirs:
	@if not exist $(DATA_DIR) mkdir $(DATA_DIR)
	@if not exist $(CONFIG_DIR) mkdir $(CONFIG_DIR)

# Generate synthetic data
generate-data:
	$(VENV_DIR)\Scripts\python $(SCRIPTS_DIR)/generate_solar_data.py

# Train LSTM model
train-model:
	$(VENV_DIR)\Scripts\python $(ML_DIR)/train_lstm.py

# Show hardware mapping
show-mapping:
	$(VENV_DIR)\Scripts\python $(SCRIPTS_DIR)/show_mapping.py

# Validate system components
validate:
	$(VENV_DIR)\Scripts\python $(SCRIPTS_DIR)/system_validator.py

# Run integration tests
test:
	$(VENV_DIR)\Scripts\python -m pytest tests/integration_test.py -v

# Run the application
run:
	$(VENV_DIR)\Scripts\python run.py

# Clean generated files
clean:
	@if exist $(DATA_DIR)/synthetic_data.csv del $(DATA_DIR)\synthetic_data.csv
	@if exist $(ML_DIR)/lstm_model.h5 del $(ML_DIR)\lstm_model.h5
# Provides targets for synthetic data generation, model training, and system validation

# Default Python interpreter
PYTHON = python

# Directories
CONFIG_DIR = config
SCRIPTS_DIR = scripts
ML_DIR = ml_model
LOGS_DIR = logs
DOCS_DIR = docs

# Files
HARDWARE_CONFIG = $(CONFIG_DIR)/hardware.json
SYNTHETIC_DATA = $(ML_DIR)/data/synthetic_solar.csv
LSTM_MODEL = $(ML_DIR)/saved_models/lstm_model_latest.h5

# Default parameters
DAYS = 30
INTERVAL = 5
SEQ_LENGTH = 24
FORECAST = 12
EPOCHS = 50

# Create necessary directories
.PHONY: init
init:
	@echo "Creating necessary directories..."
	@mkdir -p $(CONFIG_DIR) $(SCRIPTS_DIR) $(ML_DIR)/data $(ML_DIR)/saved_models $(LOGS_DIR) $(DOCS_DIR)
	@echo "Directory structure created."

# Generate synthetic solar data
.PHONY: synthetic-data
synthetic-data: init
	@echo "Generating synthetic solar data..."
	@$(PYTHON) $(SCRIPTS_DIR)/generate_solar_data.py --days $(DAYS) --interval $(INTERVAL)
	@echo "Synthetic data generated at $(SYNTHETIC_DATA)"

# Train LSTM model on synthetic data
.PHONY: train-synthetic
train-synthetic: synthetic-data
	@echo "Training LSTM model on synthetic data..."
	@$(PYTHON) $(ML_DIR)/train_lstm.py --synthetic-data $(SYNTHETIC_DATA) \
		--sequence-length $(SEQ_LENGTH) --forecast-horizon $(FORECAST) --epochs $(EPOCHS) \
		--model-name lstm_model_latest
	@echo "LSTM model training completed."

# Show hardware mapping
.PHONY: show-mapping
show-mapping:
	@echo "Displaying hardware-to-code mapping..."
	@$(PYTHON) $(SCRIPTS_DIR)/show_mapping.py
	@echo "Mapping display completed."

# Validate system components
.PHONY: validate
validate:
	@echo "Validating system components..."
	@$(PYTHON) $(SCRIPTS_DIR)/show_mapping.py --validate
	@echo "Hardware mapping validation completed."
	@echo "Checking synthetic data..."
	@test -f $(SYNTHETIC_DATA) || (echo "ERROR: Synthetic data not found. Run 'make synthetic-data' first." && exit 1)
	@echo "Synthetic data validation completed."
	@echo "Checking LSTM model..."
	@test -f $(LSTM_MODEL) || (echo "WARNING: LSTM model not found. Run 'make train-synthetic' to train the model.")
	@echo "System validation completed."

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	@rm -f $(SYNTHETIC_DATA)
	@echo "Synthetic data removed."

# Clean all generated files including models
.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all generated files..."
	@rm -rf $(ML_DIR)/saved_models/*
	@echo "All generated files removed."

# Default target
.PHONY: all
all: synthetic-data train-synthetic validate
	@echo "All tasks completed successfully."

# Help target
.PHONY: help
help:
	@echo "SolArithm Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  init              - Create necessary directories"
	@echo "  synthetic-data    - Generate synthetic solar data"
	@echo "  train-synthetic   - Train LSTM model on synthetic data"
	@echo "  show-mapping      - Display hardware-to-code mapping"
	@echo "  validate          - Validate system components"
	@echo "  clean             - Remove generated data files"
	@echo "  clean-all         - Remove all generated files including models"
	@echo "  all               - Run all tasks (default)"
	@echo "  help              - Display this help message"
	@echo ""
	@echo "Parameters:"
	@echo "  DAYS=30           - Number of days for synthetic data (default: 30)"
	@echo "  INTERVAL=5        - Sampling interval in minutes (default: 5)"
	@echo "  SEQ_LENGTH=24     - Sequence length for LSTM (default: 24)"
	@echo "  FORECAST=12       - Forecast horizon for LSTM (default: 12)"
	@echo "  EPOCHS=50         - Training epochs (default: 50)"
	@echo ""
	@echo "Example usage:"
	@echo "  make synthetic-data DAYS=60 INTERVAL=10"
	@echo "  make train-synthetic EPOCHS=100"
	@echo "  make all"