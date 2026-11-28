"""
Program 3/3: load a trained model, predict on a given dataset and evaluate
the predictions with the binary cross-entropy error function

    E = -1/N * sum[ y_n log p_n + (1 - y_n) log(1 - p_n) ]

where p_n is the predicted probability that example n is malignant (M).

    python3 app/predict.py --model saved_model.npz --dataset data/data_valid.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import MLP, BinaryCrossentropy, Dataset, Metrics  # noqa: E402


class Predictor:
    """Load a saved network and score a dataset with binary cross-entropy."""

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.loss = BinaryCrossentropy()

    def run(self, model_path, dataset_path, verbose=False, output=None):
        proba = self.model.predict_proba(self.dataset.X)
        p_malignant = proba[:, 1]
        pred = proba.argmax(axis=1)
        y = self.dataset.y
        metrics = Metrics(y, pred)

        print(f"model   : {model_path}  ({' -> '.join(map(str, self.model.layer_sizes))})")
        print(f'dataset : {dataset_path}  ({len(y)} examples)')

        if verbose:
            ids = self.dataset.frame['id']
            for i, (truth, guess, p) in enumerate(zip(y, pred, p_malignant)):
                mark = 'ok' if truth == guess else 'WRONG'
                print(f'  {ids.iloc[i]:>10}  true={Dataset.CLASSES[truth]}  pred={Dataset.CLASSES[guess]}'
                      f'  P(M)={p:.4f}  {mark}')

        print()
        print(f'binary cross-entropy : {self.loss(y, p_malignant):.4f}')
        print(f'accuracy             : {np.mean(pred == y):.4f}  ({int(np.sum(pred == y))}/{len(y)})')
        print(f'precision (M)        : {metrics["precision"]:.4f}')
        print(f'recall    (M)        : {metrics["recall"]:.4f}')
        print(f'f1-score  (M)        : {metrics["f1"]:.4f}')
        print()
        print('confusion matrix     pred B  pred M')
        print(f'             true B  {metrics["tn"]:>6}  {metrics["fp"]:>6}')
        print(f'             true M  {metrics["fn"]:>6}  {metrics["tp"]:>6}')

        if output:
            out = self.dataset.frame[['id', 'diagnosis']].copy()
            out['prediction'] = [Dataset.CLASSES[i] for i in pred]
            out['p_malignant'] = p_malignant
            out.to_csv(output, index=False)
            print(f"\n> predictions written to '{output}'")

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description='Predict with a trained multilayer perceptron.')
        parser.add_argument('--model', default='saved_model.npz', help='default: saved_model.npz')
        parser.add_argument('--dataset', default='data/data_valid.csv', help='default: data/data_valid.csv')
        parser.add_argument('--output', help='optional CSV file to write the predictions to')
        parser.add_argument('--verbose', '-v', action='store_true', help='print every prediction')
        args = parser.parse_args()

        try:
            model = MLP.load(args.model)
            dataset = Dataset.read(args.dataset)
        except (OSError, ValueError, KeyError) as e:
            sys.exit(f'error: {e}')

        cls(model, dataset).run(args.model, args.dataset, args.verbose, args.output)


if __name__ == '__main__':
    Predictor.main()
