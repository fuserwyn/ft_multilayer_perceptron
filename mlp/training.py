import csv

import numpy as np

from .losses import Metrics


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best = float('inf')
        self.best_epoch = 0
        self.wait = 0
        self._W = None
        self._b = None

    def update(self, epoch, val_loss, model):
        if val_loss < self.best:
            self.best = val_loss
            self.best_epoch = epoch
            self.wait = 0
            self._W = [W.copy() for W in model.W]
            self._b = [b.copy() for b in model.b]
            return False
        self.wait += 1
        return bool(self.patience) and self.wait >= self.patience

    def restore(self, model):
        model.W = self._W
        model.b = self._b


class History:
    def __init__(self):
        self.columns = {
            'epoch': [], 'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [],
            'precision': [], 'recall': [], 'f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
        }

    def __getitem__(self, key):
        return self.columns[key]

    def add(self, epoch, loss, val_loss, acc, val_acc, train_metrics, valid_metrics):
        self.columns['epoch'].append(epoch)
        self.columns['loss'].append(loss)
        self.columns['val_loss'].append(val_loss)
        self.columns['acc'].append(acc)
        self.columns['val_acc'].append(val_acc)
        for key in ('precision', 'recall', 'f1'):
            self.columns[key].append(train_metrics[key])
            self.columns[f'val_{key}'].append(valid_metrics[key])

    def save(self, path):
        keys = list(self.columns)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in zip(*[self.columns[k] for k in keys]):
                writer.writerow(dict(zip(keys, row)))

    @classmethod
    def load(cls, path):
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows or 'epoch' not in rows[0] or 'val_loss' not in rows[0]:
            raise ValueError(f'{path}: expected a history csv with epoch and val_loss columns')
        history = cls()
        history.columns = {k: [float(row[k]) for row in rows] for k in rows[0]}
        return history


class LearningCurves:
    def __init__(self, history, compare=None, best_epoch=None):
        self.history = history
        self.compare = compare or []
        self.best_epoch = best_epoch

    def save(self, path, show):
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = self.history['epoch']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(epochs, self.history['loss'], label='training loss')
        ax1.plot(epochs, self.history['val_loss'], '--', label='validation loss')
        ax1.set(title='Loss', xlabel='epochs', ylabel='loss')
        ax2.plot(epochs, self.history['acc'], label='training acc')
        ax2.plot(epochs, self.history['val_acc'], label='validation acc')
        ax2.set(title='Accuracy', xlabel='epochs', ylabel='accuracy')
        for other in self.compare:
            name = other.rsplit('/', 1)[-1].removesuffix('.csv')
            prev = History.load(other)
            ax1.plot(prev['epoch'], prev['val_loss'], label=f'{name} val_loss')
            if 'val_acc' in prev.columns:
                ax2.plot(prev['epoch'], prev['val_acc'], label=f'{name} val_acc')
        if self.best_epoch is not None:
            for ax in (ax1, ax2):
                ax.axvline(self.best_epoch, color='0.4', lw=0.8, ls=':', label=f'best epoch {self.best_epoch}')
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle('Learning Curves')
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        print(f"> learning curves saved to '{path}'")
        if show:
            plt.show()


class Trainer:
    def __init__(self, model, loss, learning_rate, batch_size, seed, epochs, early_stopping=0):
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.epochs = epochs
        self.early = EarlyStopping(early_stopping)
        self.history = History()

    def evaluate(self, X, Y):
        P = self.model.predict_proba(X)
        pred = P.argmax(axis=1)
        y = Y.argmax(axis=1)
        return self.loss(Y, P), float(np.mean(pred == y)), Metrics(y, pred)

    def fit(self, X_train, Y_train, X_valid, Y_valid):
        Xn_train = self.model.normalize(X_train)
        rng = np.random.default_rng(self.seed)
        width = len(str(self.epochs))

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(Xn_train))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                self.model.forward(Xn_train[batch])
                dW, db = self.model.backward(Y_train[batch])
                self.model.step(dW, db, self.learning_rate)

            loss, acc, train_m = self.evaluate(X_train, Y_train)
            val_loss, val_acc, valid_m = self.evaluate(X_valid, Y_valid)
            self.history.add(epoch, loss, val_loss, acc, val_acc, train_m, valid_m)
            print(f'epoch {epoch:0{width}d}/{self.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}'
                  f' - acc: {acc:.4f} - val_acc: {val_acc:.4f}'
                  f' - val_precision: {valid_m["precision"]:.4f} - val_recall: {valid_m["recall"]:.4f}'
                  f' - val_f1: {valid_m["f1"]:.4f}')

            if self.early.update(epoch, val_loss, self.model):
                self.early.restore(self.model)
                print(f'> early stopping at epoch {epoch}: val_loss did not improve for {self.early.wait} epochs')
                print(f'> restoring weights from epoch {self.early.best_epoch} (val_loss {self.early.best:.4f})')
                break
        return self.history
