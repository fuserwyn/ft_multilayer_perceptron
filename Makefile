# Makefile for Multilayer Perceptron Project
# All commands for running the MLP training pipeline

.PHONY: all prepare split train predict test clean help install

# Default target
all: help

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install numpy pandas matplotlib

# Step 1: Prepare and explore data
prepare:
	@echo "============================================================"
	@echo "STEP 1: DATA PREPARATION"
	@echo "============================================================"
	python3 prepare_data.py

# Step 2: Split data and normalize
split: prepare
	@echo "============================================================"
	@echo "STEP 2: DATA SPLITTING AND NORMALIZATION"
	@echo "============================================================"
	python3 split_data.py

# Step 3: Train MLP model
train: split
	@echo "============================================================"
	@echo "STEP 3: MLP TRAINING"
	@echo "============================================================"
	python3 train_mlp.py

# Step 4: Make predictions
predict: train
	@echo "============================================================"
	@echo "STEP 4: PREDICTIONS"
	@echo "============================================================"
	python3 predict.py

# Run all tests
test:
	@echo "============================================================"
	@echo "RUNNING TESTS"
	@echo "============================================================"
	python3 test_mlp.py

# Run full pipeline (prepare -> split -> train -> predict)
pipeline: predict
	@echo "============================================================"
	@echo "FULL PIPELINE COMPLETED!"
	@echo "============================================================"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f X_train.npy X_val.npy y_train.npy y_val.npy
	rm -f train_mean.npy train_std.npy
	rm -f mlp_model.npz mlp_weights.npy
	rm -f training_history.png
	rm -rf __pycache__/
	@echo "Clean completed!"

# Clean all including model
clean-all: clean
	@echo "Cleaning model files..."
	rm -f mlp_model.npz

# Help message
help:
	@echo "============================================================"
	@echo "Multilayer Perceptron - Makefile Commands"
	@echo "============================================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Install required dependencies"
	@echo "  make prepare      - Step 1: Prepare and explore data"
	@echo "  make split        - Step 2: Split data and normalize"
	@echo "  make train        - Step 3: Train MLP model"
	@echo "  make predict      - Step 4: Make predictions"
	@echo "  make test         - Run unit tests"
	@echo "  make pipeline     - Run full pipeline (all steps)"
	@echo "  make clean        - Remove generated data files"
	@echo "  make clean-all    - Remove all generated files including model"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make pipeline     - Run complete pipeline"
	@echo "  make train        - Train model (requires split step first)"
	@echo "  make test         - Run tests"
	@echo ""

