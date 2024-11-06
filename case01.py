"""
Restricted Boltzmann Machine (RBM) with MNIST Example

This code provides a complete example of a binary Restricted Boltzmann Machine (RBM) implemented in Python. 
It demonstrates the key components of an RBM, including Gibbs sampling, Contrastive Divergence (CD-1) learning,
and visualization of the learned features.

The example uses the MNIST dataset for training and evaluating the RBM's performance on a standard image 
recognition task.

Requirements:
- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn (for MNIST data loading)

Usage:
1. Install the required libraries.
2. Load and preprocess the MNIST dataset.
3. Train the RBM model.
4. Evaluate the model's performance and visualize the learned features.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

class RestrictedBoltzmannMachine:
    """
    Restricted Boltzmann Machine (RBM)
    
    Attributes:
    num_visible (int): Number of visible (input) units
    num_hidden (int): Number of hidden units
    W (np.ndarray): Weight matrix between visible and hidden units
    b (np.ndarray): Bias vector for hidden units
    c (np.ndarray): Bias vector for visible units
    """
    
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        # Initialize weights and biases to small random values
        self.W = 0.01 * np.random.randn(num_visible, num_hidden)
        self.b = np.zeros(num_hidden)
        self.c = np.zeros(num_visible)
    
    def sample_h_given_v(self, v):
        """
        Sample hidden unit states given visible unit states.
        
        Args:
        v (np.ndarray): Visible unit states
        
        Returns:
        np.ndarray: Sampled hidden unit states
        """
        h_prob = self.sigmoid(np.dot(v, self.W) + self.b)
        h_sample = (h_prob > np.random.rand(self.num_hidden)).astype(int)
        return h_sample
    
    def sample_v_given_h(self, h):
        """
        Sample visible unit states given hidden unit states.
        
        Args:
        h (np.ndarray): Hidden unit states
        
        Returns:
        np.ndarray: Sampled visible unit states
        """
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.c)
        v_sample = (v_prob > np.random.rand(self.num_visible)).astype(int)
        return v_sample
    
    def gibbs_sample(self, v):
        """
        Perform one step of Gibbs sampling.
        
        Args:
        v (np.ndarray): Initial visible unit states
        
        Returns:
        np.ndarray, np.ndarray: Final visible and hidden unit states
        """
        h = self.sample_h_given_v(v)
        v = self.sample_v_given_h(h)
        return v, h
    
    def train(self, train_data, epochs=10, lr=0.1):
        """
        Train the RBM using Contrastive Divergence (CD-1) learning.
        
        Args:
        train_data (np.ndarray): Training data matrix
        epochs (int): Number of training epochs
        lr (float): Learning rate
        """
        num_samples = train_data.shape[0]
        
        for epoch in range(epochs):
            # Positive phase
            v_pos = train_data
            h_pos = self.sample_h_given_v(v_pos)
            
            # Negative phase
            v_neg, h_neg = self.gibbs_sample(v_pos)
            
            # Update weights and biases
            self.W += lr * (np.dot(v_pos.T, h_pos) - np.dot(v_neg.T, h_neg)) / num_samples
            self.b += lr * (h_pos.mean(axis=0) - h_neg.mean(axis=0))
            self.c += lr * (v_pos.mean(axis=0) - v_neg.mean(axis=0))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1} complete.")
    
    def evaluate(self, test_data):
        """
        Evaluate the RBM's performance on the test data.
        
        Args:
        test_data (np.ndarray): Test data matrix
        
        Returns:
        float: Reconstruction error
        """
        v_rec, _ = self.gibbs_sample(test_data)
        return np.mean((test_data - v_rec) ** 2)
    
    def visualize_features(self):
        """
        Visualize the learned features (weights) of the RBM.
        """
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.W[:, i].reshape(28, 28), cmap='gray')
            ax.axis('off')
        
        plt.suptitle("Learned Features (Weights) of the RBM")
        plt.show()
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

# Example usage
def main():
    # Load and preprocess MNIST dataset
    mnist = fetch_openml('mnist_784', as_frame=False)
    train_data = mnist.data[:50000].astype(float) / 255
    test_data = mnist.data[50000:].astype(float) / 255
    
    # Initialize RBM
    rbm = RestrictedBoltzmannMachine(num_visible=784, num_hidden=500)
    
    # Train the RBM
    rbm.train(train_data, epochs=100, lr=0.1)
    
    # Evaluate the RBM
    test_error = rbm.evaluate(test_data)
    print(f"Test Reconstruction Error: {test_error:.4f}")
    
    # Visualize the learned features
    rbm.visualize_features()

if __name__ == "__main__":
    main()