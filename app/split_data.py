import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import Dataset  # noqa: E402


class DataSplitter:
    def __init__(self, valid_ratio=0.2, seed=42, out_dir='data'):
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.out_dir = out_dir

    def run(self, path):
        dataset = Dataset.read(path)
        train, valid = dataset.stratified_split(self.valid_ratio, self.seed)
        os.makedirs(self.out_dir, exist_ok=True)
        written = []
        for name, part in (('train', train), ('valid', valid)):
            dest = os.path.join(self.out_dir, f'data_{name}.csv')
            part.save(dest)
            counts = part.counts()
            print(f'{name:<5}: {len(part.frame):>3} examples  '
                  f'(M: {counts.get("M", 0)}, B: {counts.get("B", 0)})  -> {dest}')
            written.append(dest)
        return written

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description='Split the dataset into train / validation CSV files.')
        parser.add_argument('--dataset', default='data/data.csv', help='raw dataset (default: data/data.csv)')
        parser.add_argument('--valid_ratio', '--valid-ratio', type=float, default=0.2,
                            help='fraction of examples used for validation (default: 0.2)')
        parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
        parser.add_argument('--out_dir', '--out-dir', default='data', help='output directory (default: data)')
        args = parser.parse_args()

        if not 0.0 < args.valid_ratio < 1.0:
            parser.error('--valid_ratio must be in (0, 1)')

        try:
            cls(args.valid_ratio, args.seed, args.out_dir).run(args.dataset)
        except (OSError, ValueError) as e:
            sys.exit(f'error: {e}')


if __name__ == '__main__':
    DataSplitter.main()
