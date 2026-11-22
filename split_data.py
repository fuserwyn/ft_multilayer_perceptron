"""
Program 1/3: split the raw dataset into a training part and a validation part.

The split is stratified (same M/B ratio in both parts) and reproducible with --seed.
Output files keep the original raw format (no header, 32 columns), so the
normalisation is learned later, by the training program, on the training part only.

    python3 split_data.py --dataset data.csv --valid_ratio 0.2 --seed 42
"""

import argparse
import os
import sys

import numpy as np

from mlp import read_csv


def stratified_split(labels, valid_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx, valid_idx = [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_valid = int(round(len(idx) * valid_ratio))
        valid_idx.append(idx[:n_valid])
        train_idx.append(idx[n_valid:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    valid_idx = rng.permutation(np.concatenate(valid_idx))
    return train_idx, valid_idx


def main():
    parser = argparse.ArgumentParser(description='Split the dataset into train / validation CSV files.')
    parser.add_argument('--dataset', default='data.csv', help='raw dataset (default: data.csv)')
    parser.add_argument('--valid_ratio', '--valid-ratio', type=float, default=0.2,
                        help='fraction of examples used for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--out_dir', '--out-dir', default='.', help='output directory (default: .)')
    args = parser.parse_args()

    if not 0.0 < args.valid_ratio < 1.0:
        parser.error('--valid_ratio must be in (0, 1)')

    try:
        df = read_csv(args.dataset)
    except (OSError, ValueError) as e:
        sys.exit(f'error: {e}')

    train_idx, valid_idx = stratified_split(df['diagnosis'].to_numpy(), args.valid_ratio, args.seed)
    train, valid = df.iloc[train_idx], df.iloc[valid_idx]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, 'data_train.csv')
    valid_path = os.path.join(args.out_dir, 'data_valid.csv')
    train.to_csv(train_path, header=False, index=False)
    valid.to_csv(valid_path, header=False, index=False)

    for name, part, path in (('train', train, train_path), ('valid', valid, valid_path)):
        counts = part['diagnosis'].value_counts()
        print(f'{name:<5}: {len(part):>3} examples  (M: {counts.get("M", 0)}, B: {counts.get("B", 0)})'
              f'  -> {path}')


if __name__ == '__main__':
    main()
