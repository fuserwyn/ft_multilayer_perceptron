# Multilayer Perceptron

A multilayer perceptron written from scratch with numpy. It classifies breast
tumours as malignant (**M**) or benign (**B**) using the Wisconsin Breast Cancer dataset.

## Files

| file | purpose |
|---|---|
| `mlp.py` | core: activations, softmax, losses, `MLP` class (feedforward, backprop, gradient descent, save/load) |
| `split_data.py` | **program 1**: splits `data.csv` into `data_train.csv` and `data_valid.csv` (stratified, seeded) |
| `train_mlp.py` | **program 2**: trains the model, prints metrics every epoch, saves the model, shows the learning curves |
| `predict.py` | **program 3**: loads the model, predicts on a given CSV and computes the binary cross-entropy |
| `prepare_data.py` | dataset exploration (optional) |
| `test_mlp.py` | unit tests, including a numerical gradient check |

## Usage

```bash
make install        # numpy, pandas, matplotlib
make split          # or: python3 split_data.py --dataset data.csv --valid_ratio 0.2 --seed 42
make train          # or: python3 train_mlp.py
make predict        # or: python3 predict.py --model saved_model.npz --dataset data_valid.csv
make test
```

The architecture and hyper-parameters are set with arguments (at least 2 hidden layers):

```bash
python3 train_mlp.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
                     --batch_size 8 --learning_rate 0.0314 --activation sigmoid
# --activation sigmoid|tanh|relu   --weights_init heUniform|heNormal|xavierUniform
# --no_show   save the plot without opening a window
make train ARGS="--layer 32 16 --activation relu"
```

`predict.py -v` prints every prediction, and `--output preds.csv` writes them to a file.

### What `saved_model.npz` contains
- the topology: layer sizes, hidden activation, softmax output, weight initialiser, classes;
- the weights `W0..Wn` and biases `b0..bn`;
- the normalisation `mean` and `std`, computed on the **training set only**.

## Results (defaults: 30 → 24 → 24 → 2, sigmoid, lr 0.0314, batch 8, 70 epochs)

```
epoch 70/70 - loss: 0.0480 - val_loss: 0.0872 - acc: 0.9912 - val_acc: 0.9735
binary cross-entropy : 0.0872      accuracy : 0.9735 (110/113)
```

---

## Evaluation cheat sheet

Notation: a batch `X` has shape `(m, 30)`; layer `l` has `W[l]` of shape `(n_in, n_out)` and `b[l]` of shape `(1, n_out)`.

### Feedforward (`MLP.forward`)
The data flows through the network layer by layer:
```
Z[l] = A[l-1] · W[l] + b[l]        weighted sum (+ bias)
A[l] = f(Z[l])                      activation (sigmoid / tanh / relu)
P    = softmax(Z[last])             output: probability distribution over {B, M}
softmax(z)_k = e^{z_k} / Σ_j e^{z_j}     (max(z) is subtracted before exp to avoid overflow)
```
Every `Z` and `A` is cached, because backprop needs them.

### Loss function
Categorical cross-entropy: `L = -1/m Σ Σ_k Y_k log P_k`, where `Y` holds one-hot labels.
With two classes this is identical to the subject's binary cross-entropy with `p = P(M)`
(checked by `test_losses`).

### Backpropagation (`MLP.backward`)
The chain rule, applied from the output back to the input.
1. Output layer. For softmax + cross-entropy the derivative simplifies to **`dZ = (P − Y) / m`**.
2. For each layer, from the last to the first:
```
dW[l] = A[l-1]ᵀ · dZ          how much each weight contributes to the error
db[l] = Σ_batch dZ
dA    = dZ · W[l]ᵀ             pass the error down to the previous layer
dZ    = dA ⊙ f'(Z[l-1])        multiply by the activation derivative
```
`test_gradient_check` verifies the gradients against `(L(w+ε) − L(w−ε)) / 2ε`.

### Gradient descent (`MLP.step` + the loop in `train_mlp.py`)
`W ← W − lr · dW`, `b ← b − lr · db`. The gradient points towards increasing error, so we step the opposite way.
Training uses **mini-batches**: each epoch the data is shuffled and cut into batches of `batch_size`, and the weights are updated after every batch.
- If `lr` is too large, training diverges or oscillates; if it is too small, training is slow.
- An epoch is one full pass over the training set.

### Common questions
- **Why a bias?** It shifts the activation threshold; without it the separating hyperplane always passes through 0.
- **Why not initialise the weights to zero?** Every neuron in a layer would be identical and receive identical gradients, so the symmetry never breaks. He: `U(±√(6/n_in))`, Xavier: `U(±√(6/(n_in+n_out)))`.
- **Why normalise?** Features have very different scales (area ~1000, smoothness ~0.1), which unbalances the gradients. mean and std come from the training set only, otherwise information from the validation set leaks into the model.
- **Why a validation set?** To measure performance on data the model has never seen. If the training loss keeps falling while the validation loss rises, the model is overfitting.
- **Why softmax?** It turns the output into a probability distribution (non-negative, sums to 1), and combined with cross-entropy it gives the simple gradient `P − Y`.
