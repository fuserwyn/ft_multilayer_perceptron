"""
Program 3/3: load a trained model, predict on a given dataset and evaluate
the predictions with the binary cross-entropy error function

    E = -1/N * sum[ y_n log p_n + (1 - y_n) log(1 - p_n) ]

where p_n is the predicted probability that example n is malignant (M).

    python3 predict.py --model saved_model.npz --dataset data_valid.csv
"""

import argparse
import sys

import numpy as np

from mlp import CLASSES, FEATURES, MLP, binary_crossentropy, binary_metrics, read_csv


def main():
    parser = argparse.ArgumentParser(description='Predict with a trained multilayer perceptron.')
    parser.add_argument('--model', default='saved_model.npz', help='default: saved_model.npz')
    parser.add_argument('--dataset', default='data_valid.csv', help='default: data_valid.csv')
    parser.add_argument('--output', help='optional CSV file to write the predictions to')
    parser.add_argument('--verbose', '-v', action='store_true', help='print every prediction')
    args = parser.parse_args()

    try:
        model = MLP.load(args.model)
        df = read_csv(args.dataset)
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f'error: {e}')

    X = df[FEATURES].to_numpy(dtype=float)
    y = (df['diagnosis'] == 'M').to_numpy(dtype=int)

    proba = model.predict_proba(X)       # (m, 2): [P(B), P(M)]
    p_malignant = proba[:, 1]
    pred = proba.argmax(axis=1)

    print(f"model   : {args.model}  ({' -> '.join(map(str, model.layer_sizes))})")
    print(f'dataset : {args.dataset}  ({len(X)} examples)')

    if args.verbose:
        for i, (truth, guess, p) in enumerate(zip(y, pred, p_malignant)):
            mark = 'ok' if truth == guess else 'WRONG'
            print(f'  {df["id"].iloc[i]:>10}  true={CLASSES[truth]}  pred={CLASSES[guess]}'
                  f'  P(M)={p:.4f}  {mark}')

    m = binary_metrics(y, pred)
    tp, tn, fp, fn = m['tp'], m['tn'], m['fp'], m['fn']
    precision, recall, f1 = m['precision'], m['recall'], m['f1']

    print()
    print(f'binary cross-entropy : {binary_crossentropy(y, p_malignant):.4f}')
    print(f'accuracy             : {np.mean(pred == y):.4f}  ({int(np.sum(pred == y))}/{len(y)})')
    print(f'precision (M)        : {precision:.4f}')
    print(f'recall    (M)        : {recall:.4f}')
    print(f'f1-score  (M)        : {f1:.4f}')
    print()
    print('confusion matrix     pred B  pred M')
    print(f'             true B  {tn:>6}  {fp:>6}')
    print(f'             true M  {fn:>6}  {tp:>6}')

    if args.output:
        out = df[['id', 'diagnosis']].copy()
        out['prediction'] = [CLASSES[i] for i in pred]
        out['p_malignant'] = p_malignant
        out.to_csv(args.output, index=False)
        print(f"\n> predictions written to '{args.output}'")


if __name__ == '__main__':
    main()
