"""
Prediction Script
Step 4: Make predictions on new data using trained MLP
"""

import numpy as np
import pandas as pd

print("=" * 60)
print("STEP 4: PREDICTIONS")
print("=" * 60)

# Load normalization parameters
print("\n1. Loading normalization parameters...")
train_mean = np.load('train_mean.npy')
train_std = np.load('train_std.npy')
print("   Loaded train_mean and train_std")

# Load model
print("\n2. Loading trained MLP model...")
model_data = np.load('mlp_model.npz', allow_pickle=True)
# Extract weights and biases from npz file
num_layers = len([k for k in model_data.keys() if k.startswith('weights_')])
weights = [model_data[f'weights_{i}'] for i in range(num_layers)]
biases = [model_data[f'biases_{i}'] for i in range(num_layers)]
print("   Model loaded successfully")

# MLP class (same as in train_mlp.py)
class MLP:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        activations = [X]
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        
        # Output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(z)
        
        return output
    
    def predict(self, X, threshold=0.5):
        """Make predictions"""
        output = self.forward(X)
        return (output >= threshold).astype(int).flatten()
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        return self.forward(X).flatten()
    
    def compute_binary_cross_entropy(self, y_true, y_pred):
        """
        Binary cross-entropy error function as specified in the assignment:
        E = -1/N * sum[y_n * log(p_n) + (1-y_n) * log(1-p_n)]
        """
        N = len(y_true)
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y_true = y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true
        y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
        
        error = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return error

# Initialize model
mlp = MLP(weights, biases)

# Column names (same as in other scripts)
columns = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
    'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Feature columns (excluding id and diagnosis)
feature_columns = [col for col in columns if col not in ['id', 'diagnosis']]

def predict_from_csv(csv_path):
    """Make predictions from a CSV file"""
    print(f"\n3. Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, header=None, names=columns)
    
    # Extract features
    X = df[feature_columns].values
    
    print(f"   Loaded {len(X)} samples")
    
    # Normalize features
    print("   Normalizing features...")
    X_norm = (X - train_mean) / train_std
    
    # Make predictions
    print("   Making predictions...")
    predictions = mlp.predict(X_norm)
    probabilities = mlp.predict_proba(X_norm)
    
    # Create results dataframe
    results = df[['id']].copy()
    results['predicted_diagnosis'] = ['M' if p == 1 else 'B' for p in predictions]
    results['probability'] = probabilities
    results['confidence'] = np.maximum(probabilities, 1 - probabilities) * 100
    
    return results

def predict_single_sample(features):
    """
    Make prediction for a single sample
    
    Args:
        features: numpy array of shape (30,) with feature values
    
    Returns:
        dict with prediction, probability, and diagnosis
    """
    # Normalize
    features_norm = (features.reshape(1, -1) - train_mean) / train_std
    
    # Predict
    prob = mlp.predict_proba(features_norm)[0]
    pred = mlp.predict(features_norm)[0]
    
    return {
        'diagnosis': 'M' if pred == 1 else 'B',
        'probability': prob,
        'confidence': max(prob, 1 - prob) * 100
    }

# Example: Predict on validation set
print("\n4. Example: Predicting on validation set...")
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

val_predictions = mlp.predict(X_val)
val_probabilities = mlp.predict_proba(X_val)
val_accuracy = np.mean(val_predictions == y_val)

# Compute binary cross-entropy error as required by assignment
val_bce = mlp.compute_binary_cross_entropy(y_val, val_probabilities)

print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"   Binary Cross-Entropy Error: {val_bce:.4f}")
print(f"\n   Sample predictions (first 10):")
for i in range(min(10, len(val_predictions))):
    true_label = 'M' if y_val[i] == 1 else 'B'
    pred_label = 'M' if val_predictions[i] == 1 else 'B'
    prob = val_probabilities[i]
    correct = "✓" if val_predictions[i] == y_val[i] else "✗"
    print(f"      Sample {i+1}: True={true_label}, Pred={pred_label}, "
          f"Prob={prob:.3f} {correct}")

print("\n" + "=" * 60)
print("PREDICTION SCRIPT READY!")
print("=" * 60)
print("\nUsage:")
print("  - predict_from_csv('data.csv') - predict from CSV file")
print("  - predict_single_sample(features) - predict single sample")
