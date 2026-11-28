"""A short look at the raw dataset before training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import Dataset  # noqa: E402


class DatasetReport:
    """Print the shape, the class balance and a few feature statistics."""

    def __init__(self, path):
        self.path = path
        self.dataset = Dataset.read(path)

    def show(self):
        frame = self.dataset.frame
        print("=" * 60)
        print("STEP 1: DATA EXPLORATION AND PREPARATION")
        print("=" * 60)

        print(f"\n1. Loading data from {self.path}...")
        print(f"   Loaded rows: {len(frame)}, columns: {len(frame.columns)}")

        print("\n2. Checking data structure:")
        print("   First 5 rows:")
        print(frame.head())
        print("\n   Data types:")
        print(frame.dtypes)
        print("\n   Missing values:")
        missing = frame.isna().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "   No missing values!")

        print("\n3. Analyzing target variable 'diagnosis':")
        print(frame['diagnosis'].value_counts())
        print("\n   Distribution:")
        print(frame['diagnosis'].value_counts(normalize=True) * 100)

        print("\n4. Encoding diagnosis: M -> 1, B -> 0")
        print("   After encoding:")
        print(f"{self.dataset.y.sum()} malignant, {(self.dataset.y == 0).sum()} benign")

        print("\n5. Removing 'id' column (not needed for training)")
        print(f"   Remaining columns: {self.dataset.X.shape[1] + 1}")

        print("\n6. Splitting into features (X) and target variable (y)")
        print(f"   X (features): shape {self.dataset.X.shape}")
        print(f"   y (labels): shape {self.dataset.y.shape}")
        print(f"   Sample labels: {self.dataset.y[:10]}")

        print("\n7. Basic statistics on features:")
        X = self.dataset.X
        print(f"   Min values: {X.min(axis=0)[:5]}...")
        print(f"   Max values: {X.max(axis=0)[:5]}...")
        print(f"   Mean values: {X.mean(axis=0)[:5]}...")
        print(f"   Std deviations: {X.std(axis=0)[:5]}...")

        print("\n" + "=" * 60)
        print("STEP 1 COMPLETED!")
        print("=" * 60)
        print("\nNext step: train/validation split and normalization")


    @classmethod
    def main(cls):
        root = Path(__file__).resolve().parents[1]
        cls(root / 'data' / 'data.csv').show()


if __name__ == '__main__':
    DatasetReport.main()
