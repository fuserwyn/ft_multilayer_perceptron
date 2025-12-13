"""
MLP Training Script
Step 3: Implement and train Multi-Layer Perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

print("=" * 60)
print("STEP 3: MLP TRAINING")
print("=" * 60)

# Load normalized data
print("\n1. Loading normalized data...")
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

print(f"   X_train shape: {X_train.shape}")
print(f"   X_val shape: {X_val.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   y_val shape: {y_val.shape}")

# MLP Architecture
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, seed=42):
        """
        Multi-Layer Perceptron
        
        Args:
            input_size: number of input features
            hidden_sizes: list of hidden layer sizes, e.g. [64, 32]
            output_size: number of output neurons (1 for binary classification)
            learning_rate: learning rate for gradient descent
            seed: random seed for reproducibility
        """
        self.learning_rate = learning_rate
        np.random.seed(seed)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        print(f"\n   MLP Architecture:")
        print(f"      Input: {input_size}")
        for i, h in enumerate(hidden_sizes):
            print(f"      Hidden {i+1}: {h}")
        print(f"      Output: {output_size}")
        print(f"      Total parameters: {sum(w.size + b.size for w, b in zip(self.weights, self.biases))}")
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer error
        d_output = output - y.reshape(-1, 1)
        
        # Gradients for output layer
        gradients_w = []
        gradients_b = []
        
        # Output layer gradients
        dW = np.dot(self.activations[-2].T, d_output) / m
        dB = np.sum(d_output, axis=0, keepdims=True) / m
        gradients_w.append(dW)
        gradients_b.append(dB)
        
        # Backpropagate through hidden layers
        d_error = d_output
        for i in range(len(self.weights) - 2, -1, -1):
            # Error for this layer
            d_error = np.dot(d_error, self.weights[i+1].T) * self.sigmoid_derivative(self.z_values[i])
            
            # Gradients
            dW = np.dot(self.activations[i].T, d_error) / m
            dB = np.sum(d_error, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, dB)
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        """Update weights using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true.reshape(-1, 1) * np.log(y_pred) + 
                       (1 - y_true.reshape(-1, 1)) * np.log(1 - y_pred))
        return loss
    
    def predict(self, X, threshold=0.5):
        """Make predictions"""
        output = self.forward(X)
        return (output >= threshold).astype(int).flatten()
    
    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Parse command line arguments for modularity (as required by assignment)
parser = argparse.ArgumentParser(description='Train MLP for breast cancer classification')
parser.add_argument('--hidden', type=int, nargs='+', default=[64, 32],
                    help='Hidden layer sizes (default: [64, 32])')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs (default: 1000)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size for training (default: 32)')
parser.add_argument('--learning-rate', type=float, default=0.01,
                    help='Learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

args = parser.parse_args()

# Initialize MLP
print("\n2. Initializing MLP...")
input_size = X_train.shape[1]
hidden_sizes = args.hidden
output_size = 1
learning_rate = args.learning_rate

# Validate at least 2 hidden layers as required by assignment
if len(hidden_sizes) < 2:
    print("   WARNING: Assignment requires at least 2 hidden layers!")
    print(f"   Current: {len(hidden_sizes)} layer(s). Adding default second layer.")
    if len(hidden_sizes) == 1:
        hidden_sizes.append(32)
    else:
        hidden_sizes = [64, 32]

mlp = MLP(input_size, hidden_sizes, output_size, learning_rate=learning_rate, seed=args.seed)

# Training parameters
print("\n3. Training parameters:")
epochs = args.epochs
batch_size = args.batch_size
print(f"   Epochs: {epochs}")
print(f"   Batch size: {batch_size}")
print(f"   Learning rate: {learning_rate}")
print(f"   Random seed: {args.seed}")

# Training loop
print("\n4. Training MLP...")
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    epoch_train_loss = 0
    num_batches = 0
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        
        # Forward pass
        output = mlp.forward(X_batch)
        
        # Compute loss
        batch_loss = mlp.compute_loss(y_batch, output)
        epoch_train_loss += batch_loss
        num_batches += 1
        
        # Backward pass
        gradients_w, gradients_b = mlp.backward(X_batch, y_batch, output)
        
        # Update weights
        mlp.update_weights(gradients_w, gradients_b)
    
    # Average training loss
    avg_train_loss = epoch_train_loss / num_batches
    
    # Validation
    val_output = mlp.forward(X_val)
    val_loss = mlp.compute_loss(y_val, val_output)
    
    # Calculate accuracies
    train_acc = mlp.accuracy(X_train, y_train)
    val_acc = mlp.accuracy(X_val, y_val)
    
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # Print progress at each epoch (as required by assignment)
    # For readability, show full info every 10 epochs, simplified every epoch
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    else:
        # Show metrics at each epoch (simplified)
        print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} / {val_loss:.4f}, Acc: {train_acc:.4f} / {val_acc:.4f}")

# Final evaluation
print("\n5. Final evaluation:")
final_train_acc = mlp.accuracy(X_train, y_train)
final_val_acc = mlp.accuracy(X_val, y_val)
final_train_loss = mlp.compute_loss(y_train, mlp.forward(X_train))
final_val_loss = mlp.compute_loss(y_val, mlp.forward(X_val))

print(f"   Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"   Training Loss: {final_train_loss:.4f}")
print(f"   Validation Loss: {final_val_loss:.4f}")

# Save model
print("\n6. Saving model...")
# Save weights and biases as separate arrays in npz format
save_dict = {}
for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
    save_dict[f'weights_{i}'] = w
    save_dict[f'biases_{i}'] = b
np.savez('mlp_model.npz', **save_dict)
print("   Saved: mlp_model.npz")

# Plot training history
print("\n7. Plotting training history...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
ax1.plot(train_losses, label='Train Loss', alpha=0.7)
ax1.plot(val_losses, label='Validation Loss', alpha=0.7)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(train_accuracies, label='Train Accuracy', alpha=0.7)
ax2.plot(val_accuracies, label='Validation Accuracy', alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("   Saved: training_history.png")

print("\n" + "=" * 60)
print("STEP 3 COMPLETED!")
print("=" * 60)
print("\nModel trained and saved successfully!")
