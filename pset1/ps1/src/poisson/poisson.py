import numpy as np
import util
import matplotlib.pyplot as plt


def main(lr, train_path, eval_path, save_path):
  """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
  # Load training set
  x_train, y_train = util.load_dataset(train_path, add_intercept=True)
  x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

  # Train poisson model
  reg = PoissonRegression(step_size=lr)
  reg.fit(x_train, y_train)
  preds = reg.predict(x_valid)
  np.savetxt(save_path, preds)

  # plot predictions
  plt.scatter(y_valid, preds)
  plt.ylabel('Found Count')
  plt.xlabel('Expected count')
  plt.axis('equal')
  plt.savefig('poisson.png')


class PoissonRegression:
  """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

  def __init__(self,
               step_size=1e-5,
               max_iter=10000000,
               eps=1e-5,
               theta_0=None,
               verbose=True):
    """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
    self.theta = theta_0
    self.step_size = step_size
    self.max_iter = max_iter
    self.eps = eps
    self.verbose = verbose

  def y_hat(self, x):
    # The rule is y_hat = exp(theta^T * x)
    # theta is a column vector (5 x 1)
    # x is matrix (n_examples x 5)
    return np.exp(x @ self.theta)

  def gradient(self, x, y):
    # The rule is theta_j = theta_j + learning_rate * (y - y_hat) * x_j
    f = (y - self.y_hat(x))
    return x.T @ f

  def fit(self, x, y):
    """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
    if self.theta is None:
      self.theta = np.zeros(x.shape[1])
      # Fix the column dimension of theta
      self.theta = np.reshape(self.theta, (self.theta.shape[0], 1))
      y = np.reshape(y, (y.shape[0], 1))
      print(self.theta.shape, x.shape)

    iteration = 0
    while iteration < self.max_iter:
      iteration += 1
      gradient = self.gradient(x, y)
      last_theta = np.copy(self.theta)
      self.theta += self.step_size * gradient
      if np.linalg.norm(self.theta - last_theta) < self.eps:
        break
    print('Converged after %d iterations.' % iteration)

  def predict(self, x):
    """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
    # Since the mean of Poisson distribution is y_hat, we can just return y_hat
    return self.y_hat(x)


if __name__ == '__main__':
  main(lr=1e-5,
       train_path='train.csv',
       eval_path='valid.csv',
       save_path='poisson_pred.txt')
