import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import MLP, BinaryCrossentropy, Dataset, Metrics  # noqa: E402
from printer import PredictionPrinter  # noqa: E402


class Predictor:
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
        bce = self.loss(y, p_malignant)
        if output:
            out = self.dataset.frame[['id', 'diagnosis']].copy()
            out['prediction'] = [Dataset.CLASSES[i] for i in pred]
            out['p_malignant'] = p_malignant
            out.to_csv(output, index=False)
        PredictionPrinter(self.model, self.dataset, y, pred, p_malignant, metrics, bce).show(
            model_path, dataset_path, verbose, output)

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
