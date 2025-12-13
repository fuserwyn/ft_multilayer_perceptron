"""
Data exploration and preparation script
Step 1: Loading, analysis and basic preprocessing
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
print("STEP 1: DATA EXPLORATION AND PREPARATION")
print("=" * 60)

# 1. Load data
print("\n1. Loading data from data.csv...")
df = pd.read_csv('data.csv', header=None, names=columns)
print(f"   Loaded rows: {len(df)}, columns: {len(df.columns)}")

# 2. Check structure
print("\n2. Checking data structure:")
print(f"   First 5 rows:")
print(df.head())
print(f"\n   Data types:")
print(df.dtypes)
print(f"\n   Missing values:")
missing = df.isna().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   No missing values!")

# 3. Analyze target variable
print("\n3. Analyzing target variable 'diagnosis':")
print(df['diagnosis'].value_counts())
print(f"\n   Distribution:")
print(df['diagnosis'].value_counts(normalize=True) * 100)

# 4. Encode diagnosis to 0/1
print("\n4. Encoding diagnosis: M -> 1, B -> 0")
df['target'] = (df['diagnosis'] == 'M').astype(int)
print(f"   After encoding:")
print(df['target'].value_counts())

# 5. Remove id (not needed for training)
print("\n5. Removing 'id' column (not needed for training)")
df = df.drop(columns=['id'])
print(f"   Remaining columns: {len(df.columns)}")

# 6. Split into features (X) and target variable (y)
print("\n6. Splitting into features (X) and target variable (y)")
X = df.drop(columns=['diagnosis', 'target']).values  # All features except diagnosis and target
y = df['target'].values  # Target variable (0 or 1)

print(f"   X (features): shape {X.shape}")
print(f"   y (labels): shape {y.shape}")
print(f"   Sample labels: {y[:10]}")

# 7. Basic statistics on features
print("\n7. Basic statistics on features:")
print(f"   Min values: {X.min(axis=0)[:5]}...")
print(f"   Max values: {X.max(axis=0)[:5]}...")
print(f"   Mean values: {X.mean(axis=0)[:5]}...")
print(f"   Std deviations: {X.std(axis=0)[:5]}...")

print("\n" + "=" * 60)
print("STEP 1 COMPLETED!")
print("=" * 60)
print("\nNext step: train/validation split and normalization")

