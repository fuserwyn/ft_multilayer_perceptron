import numpy as np

from mlp import Dataset


class PredictionPrinter:
    def __init__(self, model, dataset, y, pred, p_malignant, metrics, bce):
        self.model = model
        self.dataset = dataset
        self.y = y
        self.pred = pred
        self.p_malignant = p_malignant
        self.metrics = metrics
        self.bce = bce

    def header(self, model_path, dataset_path):
        topology = " -> ".join(map(str, self.model.layer_sizes))
        return [
            f"model   : {model_path}  ({topology})",
            f"dataset : {dataset_path}  ({len(self.y)} examples)",
        ]

    def examples(self):
        ids = self.dataset.frame['id']
        lines = []
        for i, (truth, guess, p) in enumerate(zip(self.y, self.pred, self.p_malignant)):
            mark = 'ok' if truth == guess else 'WRONG'
            lines.append(
                f'  {ids.iloc[i]:>10}  true={Dataset.CLASSES[truth]}  pred={Dataset.CLASSES[guess]}'
                f'  P(M)={p:.4f}  {mark}'
            )
        return lines

    def scores(self):
        correct = int(np.sum(self.pred == self.y))
        accuracy = np.mean(self.pred == self.y)
        m = self.metrics
        return [
            "",
            f'binary cross-entropy : {self.bce:.4f}',
            f'accuracy             : {accuracy:.4f}  ({correct}/{len(self.y)})',
            f'precision (M)        : {m["precision"]:.4f}',
            f'recall    (M)        : {m["recall"]:.4f}',
            f'f1-score  (M)        : {m["f1"]:.4f}',
            "",
            'confusion matrix     pred B  pred M',
            f'             true B  {m["tn"]:>6}  {m["fp"]:>6}',
            f'             true M  {m["fn"]:>6}  {m["tp"]:>6}',
        ]

    def show(self, model_path, dataset_path, verbose=False, output=None):
        lines = self.header(model_path, dataset_path)
        if verbose:
            lines.extend(self.examples())
        lines.extend(self.scores())
        if output:
            lines.append("")
            lines.append(f"> predictions written to '{output}'")
        print("\n".join(lines))
