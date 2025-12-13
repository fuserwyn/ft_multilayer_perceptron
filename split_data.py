"""
Data splitting and normalization script
Step 2: Split dataset into train/validation and normalize features
"""

import pandas as pd
import numpy as np

# Column names according to the standard Wisconsin Breast Cancer dataset
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

print("=" * 60)
print("STEP 2: DATA SPLITTING AND NORMALIZATION")
print("=" * 60)

# Load and prepare data
print("\n1. Loading data from data.csv...")
df = pd.read_csv('data.csv', header=None, names=columns)

# Encode diagnosis: M -> 1, B -> 0
df['target'] = (df['diagnosis'] == 'M').astype(int)

# Separate features and target
X = df.drop(columns=['id', 'diagnosis', 'target']).values
y = df['target'].values

print(f"   Original data shape: X={X.shape}, y={y.shape}")

# Split into train and validation sets
print("\n2. Splitting into train/validation sets (80/20)...")
SEED = 42  # For reproducibility
test_size = 0.2

# Manual stratified split to preserve class distribution
np.random.seed(SEED)
indices = np.arange(len(X))

# Separate indices by class for stratified split
class_0_indices = indices[y == 0]
class_1_indices = indices[y == 1]

# Shuffle each class separately
np.random.shuffle(class_0_indices)
np.random.shuffle(class_1_indices)

# Calculate split sizes for each class
n_val_0 = int(len(class_0_indices) * test_size)
n_val_1 = int(len(class_1_indices) * test_size)

# Split indices
val_indices = np.concatenate([
    class_0_indices[:n_val_0],
    class_1_indices[:n_val_1]
])
train_indices = np.concatenate([
    class_0_indices[n_val_0:],
    class_1_indices[n_val_1:]
])

# Shuffle final indices
np.random.shuffle(val_indices)
np.random.shuffle(train_indices)

# Create splits
X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y[train_indices], y[val_indices]

print(f"   Train set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"   Validation set: X_val={X_val.shape}, y_val={y_val.shape}")

# Check class distribution in both sets
print(f"\n   Train labels distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"      Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")

print(f"   Validation labels distribution:")
unique, counts = np.unique(y_val, return_counts=True)
for label, count in zip(unique, counts):
    print(f"      Class {label}: {count} samples ({count/len(y_val)*100:.1f}%)")

# Normalize features using StandardScaler (z-score normalization)
print("\n3. Normalizing features using StandardScaler...")
print("   Formula: z = (x - mean) / std")

# Calculate mean and std from TRAIN set only (important!)
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# Avoid division by zero
train_std = np.where(train_std == 0, 1, train_std)

print(f"   Train set - Mean range: [{train_mean.min():.2f}, {train_mean.max():.2f}]")
print(f"   Train set - Std range: [{train_std.min():.2f}, {train_std.max():.2f}]")

# Normalize train set
X_train_norm = (X_train - train_mean) / train_std

# Normalize validation set using SAME mean and std from train set
X_val_norm = (X_val - train_mean) / train_std

print(f"\n   After normalization:")
print(f"   X_train_norm: mean={X_train_norm.mean(axis=0).mean():.6f}, std={X_train_norm.std(axis=0).mean():.6f}")
print(f"   X_val_norm: mean={X_val_norm.mean(axis=0).mean():.6f}, std={X_val_norm.std(axis=0).mean():.6f}")

# Save normalized data
print("\n4. Saving normalized data...")
np.save('X_train.npy', X_train_norm)
np.save('X_val.npy', X_val_norm)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

# Also save normalization parameters (needed for prediction later)
np.save('train_mean.npy', train_mean)
np.save('train_std.npy', train_std)

print("   Saved files:")
print("      - X_train.npy (normalized training features)")
print("      - X_val.npy (normalized validation features)")
print("      - y_train.npy (training labels)")
print("      - y_val.npy (validation labels)")
print("      - train_mean.npy (mean for normalization)")
print("      - train_std.npy (std for normalization)")

print("\n" + "=" * 60)
print("STEP 2 COMPLETED!")
print("=" * 60)
print("\nNext step: Implement MLP architecture and training")

