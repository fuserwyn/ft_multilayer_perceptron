"""
Unit tests for MLP implementation
Tests for forward propagation, backpropagation, and gradient descent
"""

import numpy as np
import sys
import os

# Add current directory to path to import MLP class
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MLP class from train_mlp.py
# We need to suppress the print statements and training execution
import io
import contextlib

# Suppress output during import
f = io.StringIO()
with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    # Import will execute the script, but we suppress output
    import train_mlp
    MLP = train_mlp.MLP


def test_sigmoid():
    """Test sigmoid activation function"""
    print("Testing sigmoid function...")
    mlp = MLP(2, [2], 1, seed=42)
    
    # Test sigmoid at 0 should be 0.5
    assert abs(mlp.sigmoid(0) - 0.5) < 1e-6, "Sigmoid(0) should be 0.5"
    
    # Test sigmoid at large positive value should be close to 1
    assert mlp.sigmoid(10) > 0.99, "Sigmoid(10) should be close to 1"
    
    # Test sigmoid at large negative value should be close to 0
    assert mlp.sigmoid(-10) < 0.01, "Sigmoid(-10) should be close to 0"
    
    # Test sigmoid derivative
    x = np.array([0, 1, -1])
    sig = mlp.sigmoid(x)
    sig_der = mlp.sigmoid_derivative(x)
    expected_der = sig * (1 - sig)
    assert np.allclose(sig_der, expected_der), "Sigmoid derivative incorrect"
    
    print("✓ Sigmoid tests passed")


def test_mlp_initialization():
    """Test MLP initialization"""
    print("\nTesting MLP initialization...")
    
    mlp = MLP(input_size=5, hidden_sizes=[4, 3], output_size=1, seed=42)
    
    # Check number of weight matrices
    assert len(mlp.weights) == 3, "Should have 3 weight matrices (input->hidden1, hidden1->hidden2, hidden2->output)"
    assert len(mlp.biases) == 3, "Should have 3 bias vectors"
    
    # Check dimensions
    assert mlp.weights[0].shape == (5, 4), "First weight matrix should be (5, 4)"
    assert mlp.weights[1].shape == (4, 3), "Second weight matrix should be (4, 3)"
    assert mlp.weights[2].shape == (3, 1), "Third weight matrix should be (3, 1)"
    
    assert mlp.biases[0].shape == (1, 4), "First bias should be (1, 4)"
    assert mlp.biases[1].shape == (1, 3), "Second bias should be (1, 3)"
    assert mlp.biases[2].shape == (1, 1), "Third bias should be (1, 1)"
    
    print("✓ MLP initialization tests passed")


def test_forward_propagation():
    """Test forward propagation"""
    print("\nTesting forward propagation...")
    
    mlp = MLP(input_size=3, hidden_sizes=[2], output_size=1, seed=42)
    
    # Create simple input
    X = np.array([[1.0, 2.0, 3.0]])
    
    # Forward pass
    output = mlp.forward(X)
    
    # Check output shape
    assert output.shape == (1, 1), f"Output shape should be (1, 1), got {output.shape}"
    
    # Check output is in valid range [0, 1] (sigmoid output)
    assert 0 <= output[0, 0] <= 1, "Output should be in [0, 1] range"
    
    # Check activations are stored
    assert len(mlp.activations) == 3, "Should have 3 activation layers (input, hidden, output)"
    assert mlp.activations[0].shape == (1, 3), "Input activation shape incorrect"
    assert mlp.activations[1].shape == (1, 2), "Hidden activation shape incorrect"
    assert mlp.activations[2].shape == (1, 1), "Output activation shape incorrect"
    
    # Check z_values are stored
    assert len(mlp.z_values) == 2, "Should have 2 z_value layers (hidden, output)"
    
    # Test with batch
    X_batch = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output_batch = mlp.forward(X_batch)
    assert output_batch.shape == (2, 1), "Batch output shape should be (2, 1)"
    
    print("✓ Forward propagation tests passed")


def test_backward_propagation():
    """Test backward propagation"""
    print("\nTesting backward propagation...")
    
    mlp = MLP(input_size=3, hidden_sizes=[2], output_size=1, seed=42)
    
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1])
    
    # Forward pass
    output = mlp.forward(X)
    
    # Backward pass
    gradients_w, gradients_b = mlp.backward(X, y, output)
    
    # Check gradients structure
    assert len(gradients_w) == 2, "Should have 2 weight gradients"
    assert len(gradients_b) == 2, "Should have 2 bias gradients"
    
    # Check gradient shapes match weight shapes
    assert gradients_w[0].shape == mlp.weights[0].shape, "Gradient shape should match weight shape"
    assert gradients_w[1].shape == mlp.weights[1].shape, "Gradient shape should match weight shape"
    assert gradients_b[0].shape == mlp.biases[0].shape, "Gradient shape should match bias shape"
    assert gradients_b[1].shape == mlp.biases[1].shape, "Gradient shape should match bias shape"
    
    # Test with batch
    X_batch = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_batch = np.array([1, 0])
    output_batch = mlp.forward(X_batch)
    gradients_w_batch, gradients_b_batch = mlp.backward(X_batch, y_batch, output_batch)
    
    assert gradients_w_batch[0].shape == mlp.weights[0].shape, "Batch gradient shape should match weight shape"
    
    print("✓ Backward propagation tests passed")


def test_gradient_descent():
    """Test gradient descent weight update"""
    print("\nTesting gradient descent...")
    
    mlp = MLP(input_size=3, hidden_sizes=[2], output_size=1, learning_rate=0.1, seed=42)
    
    # Save initial weights
    initial_weights = [w.copy() for w in mlp.weights]
    initial_biases = [b.copy() for b in mlp.biases]
    
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1])
    
    # Forward and backward pass
    output = mlp.forward(X)
    gradients_w, gradients_b = mlp.backward(X, y, output)
    
    # Update weights
    mlp.update_weights(gradients_w, gradients_b)
    
    # Check that weights changed
    for i in range(len(mlp.weights)):
        assert not np.allclose(mlp.weights[i], initial_weights[i]), "Weights should have changed"
        assert not np.allclose(mlp.biases[i], initial_biases[i]), "Biases should have changed"
    
    # Check update direction (weights should decrease if gradient is positive)
    # This is a simplified check
    print("✓ Gradient descent tests passed")


def test_loss_function():
    """Test binary cross-entropy loss"""
    print("\nTesting loss function...")
    
    mlp = MLP(input_size=2, hidden_sizes=[2], output_size=1, seed=42)
    
    # Test perfect prediction (y_pred = y_true)
    y_true = np.array([1, 0, 1])
    y_pred = np.array([[1.0], [0.0], [1.0]])
    loss = mlp.compute_loss(y_true, y_pred)
    
    # Loss should be very small for perfect predictions
    assert loss < 0.01, "Loss should be very small for perfect predictions"
    
    # Test worst prediction (y_pred = 1 - y_true)
    y_pred_worst = np.array([[0.0], [1.0], [0.0]])
    loss_worst = mlp.compute_loss(y_true, y_pred_worst)
    
    # Worst prediction should have higher loss
    assert loss_worst > loss, "Worst prediction should have higher loss"
    
    # Test loss is always positive
    y_pred_random = np.random.rand(3, 1)
    loss_random = mlp.compute_loss(y_true, y_pred_random)
    assert loss_random > 0, "Loss should always be positive"
    
    print("✓ Loss function tests passed")


def test_predict():
    """Test prediction function"""
    print("\nTesting prediction function...")
    
    mlp = MLP(input_size=3, hidden_sizes=[2], output_size=1, seed=42)
    
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    predictions = mlp.predict(X)
    
    # Check predictions are binary (0 or 1)
    assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"
    assert predictions.shape == (2,), "Predictions shape should be (2,)"
    
    # Test threshold
    predictions_high_threshold = mlp.predict(X, threshold=0.9)
    predictions_low_threshold = mlp.predict(X, threshold=0.1)
    
    # Higher threshold should give fewer positive predictions
    assert np.sum(predictions_high_threshold) <= np.sum(predictions_low_threshold), \
        "Higher threshold should give fewer positive predictions"
    
    print("✓ Prediction tests passed")


def test_accuracy():
    """Test accuracy calculation"""
    print("\nTesting accuracy function...")
    
    mlp = MLP(input_size=3, hidden_sizes=[2], output_size=1, seed=42)
    
    # Create simple dataset
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array([1, 0, 1])
    
    accuracy = mlp.accuracy(X, y)
    
    # Accuracy should be between 0 and 1
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    
    print("✓ Accuracy tests passed")


def test_training_step():
    """Test complete training step (forward + backward + update)"""
    print("\nTesting complete training step...")
    
    mlp = MLP(input_size=3, hidden_sizes=[2], output_size=1, learning_rate=0.01, seed=42)
    
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([1, 0])
    
    # Initial loss
    initial_output = mlp.forward(X)
    initial_loss = mlp.compute_loss(y, initial_output)
    
    # Training step
    output = mlp.forward(X)
    gradients_w, gradients_b = mlp.backward(X, y, output)
    mlp.update_weights(gradients_w, gradients_b)
    
    # Loss after one step (should generally decrease, but not guaranteed)
    new_output = mlp.forward(X)
    new_loss = mlp.compute_loss(y, new_output)
    
    # Loss should be a valid number
    assert not np.isnan(new_loss), "Loss should not be NaN"
    assert not np.isinf(new_loss), "Loss should not be infinite"
    
    print("✓ Training step tests passed")


def test_reproducibility():
    """Test that same seed gives same initialization"""
    print("\nTesting reproducibility...")
    
    mlp1 = MLP(input_size=3, hidden_sizes=[2], output_size=1, seed=42)
    mlp2 = MLP(input_size=3, hidden_sizes=[2], output_size=1, seed=42)
    
    # Check weights are the same
    for w1, w2 in zip(mlp1.weights, mlp2.weights):
        assert np.allclose(w1, w2), "Same seed should give same weights"
    
    for b1, b2 in zip(mlp1.biases, mlp2.biases):
        assert np.allclose(b1, b2), "Same seed should give same biases"
    
    print("✓ Reproducibility tests passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("RUNNING MLP TESTS")
    print("=" * 60)
    
    try:
        test_sigmoid()
        test_mlp_initialization()
        test_forward_propagation()
        test_backward_propagation()
        test_gradient_descent()
        test_loss_function()
        test_predict()
        test_accuracy()
        test_training_step()
        test_reproducibility()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

