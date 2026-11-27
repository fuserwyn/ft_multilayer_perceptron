"""
Program 2/3: train the multilayer perceptron with backpropagation and
mini-batch gradient descent, then save the model (topology + weights +
normalisation) and display the learning curves.

    python3 train_mlp.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
                         --batch_size 8 --learning_rate 0.0314
"""

import argparse
import csv
import sys

import numpy as np

from mlp import (ACTIVATIONS, LOSSES, MLP, OPTIMIZERS, WEIGHT_INITS, binary_metrics, load_xy, one_hot)


def parse_args():
    p = argparse.ArgumentParser(description='Train the multilayer perceptron.')
    p.add_argument('--train', default='data_train.csv', help='training set (default: data_train.csv)')
    p.add_argument('--valid', default='data_valid.csv', help='validation set (default: data_valid.csv)')
    p.add_argument('--layer', type=int, nargs='+', default=[24, 24],
                   help='hidden layer sizes, at least two (default: 24 24)')
    p.add_argument('--epochs', type=int, default=70, help='default: 70')
    p.add_argument('--loss', choices=list(LOSSES), default='categoricalCrossentropy')
    p.add_argument('--batch_size', '--batch-size', type=int, default=8, help='default: 8')
    p.add_argument('--learning_rate', '--learning-rate', type=float, default=0.0314,
                   help='default: 0.0314')
    p.add_argument('--activation', choices=list(ACTIVATIONS), default='sigmoid',
                   help='hidden layers activation (default: sigmoid)')
    p.add_argument('--weights_init', '--weights-init', choices=WEIGHT_INITS, default='heUniform')
    p.add_argument('--optimizer', choices=OPTIMIZERS, default='sgd',
                   help='sgd is plain gradient descent; adam is the adaptive bonus (default: sgd)')
    p.add_argument('--early_stopping', type=int, default=0, metavar='PATIENCE',
                   help='stop when val_loss stops improving for PATIENCE epochs, and restore the best weights (default: off)')
    p.add_argument('--seed', type=int, default=42, help='seed for weights init and shuffling')
    p.add_argument('--model', default='saved_model.npz', help='output model file')
    p.add_argument('--plot', default='learning_curves.png', help='output image of the learning curves')
    p.add_argument('--history', default='training_history.csv', help='csv file of the metrics at every epoch')
    p.add_argument('--compare', nargs='*', default=[], metavar='CSV',
                   help='earlier history files to draw on the same learning curves')
    p.add_argument('--no_show', '--no-show', action='store_true',
                   help='save the learning curves without opening a window')
    args = p.parse_args()

    if len(args.layer) < 2:
        p.error('--layer: the network needs at least two hidden layers')
    if any(n <= 0 for n in args.layer):
        p.error('--layer: sizes must be positive')
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        p.error('--epochs, --batch_size and --learning_rate must be positive')
    if args.early_stopping < 0:
        p.error('--early_stopping patience must be >= 0 (0 disables it)')
    return args


def evaluate(model, X, Y, loss_fn):
    P = model.predict_proba(X)
    pred = P.argmax(axis=1)
    y = Y.argmax(axis=1)
    metrics = binary_metrics(y, pred)
    return loss_fn(Y, P), float(np.mean(pred == y)), metrics


def save_history(history, path):
    keys = list(history)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in zip(*[history[k] for k in keys]):
            writer.writerow(dict(zip(keys, row)))


def load_history(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'epoch' not in rows[0] or 'val_loss' not in rows[0]:
        raise ValueError(f'{path}: expected a history csv with epoch and val_loss columns')
    return {k: [float(row[k]) for row in rows] for k in rows[0]}


def plot_curves(history, path, show, compare=None, best_epoch=None):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = history['epoch']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history['loss'], label='training loss')
    ax1.plot(epochs, history['val_loss'], '--', label='validation loss')
    ax1.set(title='Loss', xlabel='epochs', ylabel='loss')
    ax2.plot(epochs, history['acc'], label='training acc')
    ax2.plot(epochs, history['val_acc'], label='validation acc')
    ax2.set(title='Accuracy', xlabel='epochs', ylabel='accuracy')
    for other in compare or []:
        name = other.rsplit('/', 1)[-1].removesuffix('.csv')
        prev = load_history(other)
        ax1.plot(prev['epoch'], prev['val_loss'], label=f'{name} val_loss')
        if 'val_acc' in prev:
            ax2.plot(prev['epoch'], prev['val_acc'], label=f'{name} val_acc')
    if best_epoch is not None:
        for ax in (ax1, ax2):
            ax.axvline(best_epoch, color='0.4', lw=0.8, ls=':', label=f'best epoch {best_epoch}')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle('Learning Curves')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"> learning curves saved to '{path}'")
    if show:
        plt.show()


def main():
    args = parse_args()

    try:
        X_train, y_train = load_xy(args.train)
        X_valid, y_valid = load_xy(args.valid)
    except (OSError, ValueError) as e:
        sys.exit(f'error: {e}\n(run split_data.py first)')

    Y_train, Y_valid = one_hot(y_train), one_hot(y_valid)
    print(f'x_train shape : {X_train.shape}')
    print(f'x_valid shape : {X_valid.shape}')

    if args.optimizer == 'adam' and args.learning_rate > 0.01:
        print(f'warning: Adam is usually stable around --learning_rate 0.001 '
              f'(got {args.learning_rate})')

    layer_sizes = [X_train.shape[1]] + args.layer + [2]
    model = MLP(layer_sizes, args.activation, args.weights_init, args.seed, args.optimizer)
    model.fit_normalization(X_train)                 # statistics from the training set only
    Xn_train = model.normalize(X_train)
    loss_fn = LOSSES[args.loss]

    print(f'network       : {" -> ".join(map(str, layer_sizes))}  '
          f'({args.activation} hidden, softmax output, {args.weights_init}, {args.optimizer})')

    rng = np.random.default_rng(args.seed)
    history = {
        'epoch': [], 'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [],
        'precision': [], 'recall': [], 'f1': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
    }
    width = len(str(args.epochs))
    best_loss, best_epoch, wait = float('inf'), 0, 0
    best_W = best_b = None

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(Xn_train))       # new shuffle every epoch
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            model.forward(Xn_train[batch])           # feedforward
            dW, db = model.backward(Y_train[batch])  # backpropagation
            model.step(dW, db, args.learning_rate)   # gradient descent or Adam

        loss, acc, train_m = evaluate(model, X_train, Y_train, loss_fn)
        val_loss, val_acc, valid_m = evaluate(model, X_valid, Y_valid, loss_fn)
        history['epoch'].append(epoch)
        for key, value in zip(('loss', 'val_loss', 'acc', 'val_acc'), (loss, val_loss, acc, val_acc)):
            history[key].append(value)
        for key in ('precision', 'recall', 'f1'):
            history[key].append(train_m[key])
            history[f'val_{key}'].append(valid_m[key])
        print(f'epoch {epoch:0{width}d}/{args.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}'
              f' - acc: {acc:.4f} - val_acc: {val_acc:.4f}'
              f' - val_precision: {valid_m["precision"]:.4f} - val_recall: {valid_m["recall"]:.4f}'
              f' - val_f1: {valid_m["f1"]:.4f}')

        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_W = [W.copy() for W in model.W]
            best_b = [b.copy() for b in model.b]
        else:
            wait += 1
        if args.early_stopping and wait >= args.early_stopping:
            model.W, model.b = best_W, best_b
            print(f'> early stopping at epoch {epoch}: val_loss did not improve for {wait} epochs')
            print(f'> restoring weights from epoch {best_epoch} (val_loss {best_loss:.4f})')
            break

    model.save(args.model, extra={k: v for k, v in vars(args).items()
                                  if k in ('epochs', 'loss', 'batch_size', 'learning_rate', 'seed',
                                           'optimizer', 'early_stopping')})
    print(f"> saving model '{args.model}' to disk...")
    try:
        save_history(history, args.history)
    except OSError as e:
        sys.exit(f'error: {e}')
    print(f"> metrics history saved to '{args.history}'")
    try:
        plot_curves(history, args.plot, show=not args.no_show, compare=args.compare,
                    best_epoch=best_epoch if args.early_stopping else None)
    except (OSError, ValueError) as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()
