import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegressionArtist:
    """A formal approach to logistic regression modeling and visualization."""

    def __init__(self, data_path='data', output_dir='output', learning_rate=0.1, iterations=1000):
        """Initialize the logistic regression model with specified parameters."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.plt_style = 'seaborn-v0_8-darkgrid'
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_and_prepare_data(self):
        """Load and normalize data."""
        X = pd.read_csv(f'logisticX.csv', header=None).values
        y = pd.read_csv(f'logisticY.csv', header=None).values.flatten()

        # Normalize the data
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        return X, y

    def _sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y, theta):
        """Compute the cost function."""
        m = len(y)
        predictions = self._sigmoid(X @ theta)
        cost = -1 / m * np.sum(
            y * np.log(predictions + 1e-8) +
            (1 - y) * np.log(1 - predictions + 1e-8)
        )
        return cost

    def train_model(self):
        """Train the logistic regression model using gradient descent."""
        X, y = self._load_and_prepare_data()
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        theta = np.zeros(X_with_bias.shape[1])

        cost_history = []
        for _ in range(self.iterations):
            predictions = self._sigmoid(X_with_bias @ theta)
            gradient = X_with_bias.T @ (predictions - y) / len(y)
            theta -= self.learning_rate * gradient
            cost_history.append(self._cost_function(X_with_bias, y, theta))

        return theta, cost_history

    def _create_plots(self, theta, cost_history):
        """Create and save plots to the specified directory."""
        plt.style.use(self.plt_style)

        # Cost vs Iterations
        plt.figure(figsize=(12, 6))
        plt.plot(cost_history[:50], linewidth=3, color='#1E90FF', alpha=0.7)
        plt.title('Cost Function', fontsize=16, fontweight='bold')
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.fill_between(range(50), cost_history[:50], color='#87CEFA', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cost_journey.png'), dpi=300)
        plt.close()

        # Decision Boundary
        X, y = self._load_and_prepare_data()
        plt.figure(figsize=(12, 8))

        # Scatter plot
        plt.scatter(X[y == 0, 0], X[y == 0, 1],
                    c='#3498db', label='Class 0',
                    alpha=0.7, edgecolor='white', s=100)
        plt.scatter(X[y == 1, 0], X[y == 1, 1],
                    c='#e74c3c', label='Class 1',
                    alpha=0.7, edgecolor='white', s=100)

        # Decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        grid_points = np.column_stack([np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()])
        Z = self._sigmoid(grid_points @ theta)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                     colors=['#3498db', '#95a5a6', '#e74c3c'],
                     alpha=0.2)
        plt.contour(xx, yy, Z, levels=[0.5], colors='#2c3e50',
                    linestyles='--', linewidths=3)

        plt.title('Decision Boundary', fontsize=16, fontweight='bold')
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'decision_boundary.png'), dpi=300)
        plt.close()


# Create the logistic regression model and generate plots
ml_artist = LogisticRegressionArtist()
theta, cost_history = ml_artist.train_model()
ml_artist._create_plots(theta, cost_history)
