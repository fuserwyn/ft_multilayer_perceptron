"""
Program 2/3: train the multilayer perceptron with backpropagation and
mini-batch gradient descent, then save the model (topology + weights +
normalisation) and display the learning curves.

    python3 train_mlp.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
                         --batch_size 8 --learning_rate 0.0314
"""

import argparse
import sys

import numpy as np

from mlp import (ACTIVATIONS, LOSSES, MLP, WEIGHT_INITS, load_xy, one_hot)


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
    p.add_argument('--seed', type=int, default=42, help='seed for weights init and shuffling')
    p.add_argument('--model', default='saved_model.npz', help='output model file')
    p.add_argument('--plot', default='learning_curves.png', help='output image of the learning curves')
    p.add_argument('--no_show', '--no-show', action='store_true',
                   help='save the learning curves without opening a window')
    args = p.parse_args()

    if len(args.layer) < 2:
        p.error('--layer: the network needs at least two hidden layers')
    if any(n <= 0 for n in args.layer):
        p.error('--layer: sizes must be positive')
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        p.error('--epochs, --batch_size and --learning_rate must be positive')
    return args


def evaluate(model, X, Y, loss_fn):
    P = model.predict_proba(X)
    return loss_fn(Y, P), float(np.mean(P.argmax(axis=1) == Y.argmax(axis=1)))


def plot_curves(history, path, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history['loss'], label='training loss')
    ax1.plot(epochs, history['val_loss'], '--', label='validation loss')
    ax1.set(title='Loss', xlabel='epochs', ylabel='loss')
    ax2.plot(epochs, history['acc'], label='training acc')
    ax2.plot(epochs, history['val_acc'], label='validation acc')
    ax2.set(title='Accuracy', xlabel='epochs', ylabel='accuracy')
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

    layer_sizes = [X_train.shape[1]] + args.layer + [2]
    model = MLP(layer_sizes, args.activation, args.weights_init, args.seed)
    model.fit_normalization(X_train)                 # statistics from the training set only
    Xn_train = model.normalize(X_train)
    loss_fn = LOSSES[args.loss]

    print(f'network       : {" -> ".join(map(str, layer_sizes))}  '
          f'({args.activation} hidden, softmax output, {args.weights_init})')

    rng = np.random.default_rng(args.seed)
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    width = len(str(args.epochs))

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(Xn_train))       # new shuffle every epoch
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            model.forward(Xn_train[batch])           # feedforward
            dW, db = model.backward(Y_train[batch])  # backpropagation
            model.step(dW, db, args.learning_rate)   # gradient descent

        loss, acc = evaluate(model, X_train, Y_train, loss_fn)
        val_loss, val_acc = evaluate(model, X_valid, Y_valid, loss_fn)
        for key, value in zip(history, (loss, val_loss, acc, val_acc)):
            history[key].append(value)
        print(f'epoch {epoch:0{width}d}/{args.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}'
              f' - acc: {acc:.4f} - val_acc: {val_acc:.4f}')

    model.save(args.model, extra={k: v for k, v in vars(args).items()
                                  if k in ('epochs', 'loss', 'batch_size', 'learning_rate', 'seed')})
    print(f"> saving model '{args.model}' to disk...")
    plot_curves(history, args.plot, show=not args.no_show)


if __name__ == '__main__':
    main()
