"""
Program 2/3: train the multilayer perceptron with backpropagation and
mini-batch gradient descent, then save the model (topology + weights +
normalisation) and display the learning curves.

    python3 app/train_mlp.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
                             --batch_size 8 --learning_rate 0.0314
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import ACTIVATIONS, LOSSES, MLP, OPTIMIZERS, WEIGHT_INITS, Dataset, LearningCurves, Trainer  # noqa: E402


class TrainingProgram:
    """Build a network from the arguments, train it, save it, draw the curves."""

    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        try:
            train = Dataset.read(args.train)
            valid = Dataset.read(args.valid)
        except (OSError, ValueError) as e:
            sys.exit(f'error: {e}\n(run app/split_data.py first)')

        X_train, y_train = train.X, train.y
        X_valid, y_valid = valid.X, valid.y
        Y_train, Y_valid = Dataset.encode(y_train), Dataset.encode(y_valid)
        print(f'x_train shape : {X_train.shape}')
        print(f'x_valid shape : {X_valid.shape}')

        if args.optimizer == 'adam' and args.learning_rate > 0.01:
            print(f'warning: Adam is usually stable around --learning_rate 0.001 '
                  f'(got {args.learning_rate})')

        layer_sizes = [X_train.shape[1]] + args.layer + [2]
        model = MLP(layer_sizes, args.activation, args.weights_init, args.seed, args.optimizer)
        model.fit_normalization(X_train)
        print(f'network       : {" -> ".join(map(str, layer_sizes))}  '
              f'({args.activation} hidden, softmax output, {args.weights_init}, {args.optimizer})')

        trainer = Trainer(model, LOSSES[args.loss], args.learning_rate, args.batch_size,
                          args.seed, args.epochs, args.early_stopping)
        history = trainer.fit(X_train, Y_train, X_valid, Y_valid)

        model.save(args.model, extra={k: v for k, v in vars(args).items()
                                      if k in ('epochs', 'loss', 'batch_size', 'learning_rate', 'seed',
                                               'optimizer', 'early_stopping')})
        print(f"> saving model '{args.model}' to disk...")
        try:
            history.save(args.history)
        except OSError as e:
            sys.exit(f'error: {e}')
        print(f"> metrics history saved to '{args.history}'")
        best = trainer.early.best_epoch if args.early_stopping else None
        try:
            LearningCurves(history, args.compare, best).save(args.plot, show=not args.no_show)
        except (OSError, ValueError) as e:
            sys.exit(f'error: {e}')

    @classmethod
    def parse(cls):
        p = argparse.ArgumentParser(description='Train the multilayer perceptron.')
        p.add_argument('--train', default='data/data_train.csv', help='training set (default: data/data_train.csv)')
        p.add_argument('--valid', default='data/data_valid.csv', help='validation set (default: data/data_valid.csv)')
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
        p.add_argument('--optimizer', choices=list(OPTIMIZERS), default='sgd',
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
        return cls(args)

    @classmethod
    def main(cls):
        cls.parse().run()


if __name__ == '__main__':
    TrainingProgram.main()
