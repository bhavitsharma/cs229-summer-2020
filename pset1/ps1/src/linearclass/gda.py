import numpy as np
import util


def main(train_path, valid_path, plot_path, save_path):
  """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
  # Load dataset
  x_train, y_train = util.load_dataset(train_path, add_intercept=False)
  x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)

  # *** START CODE HERE ***
  classifier = GDA()
  classifier.fit(x_train, y_train)
  util.plot(x_valid, y_valid, classifier.theta, plot_path)


class GDA:
  """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

  def __init__(self,
               step_size=0.01,
               max_iter=10000,
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
    self.phi = None
    self.mu_0 = None
    self.mu_1 = None
    self.sigma = None

  def fit(self, x, y):
    """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
    self.phi = np.mean(y)
    self.mu_0 = np.mean(x[y == 0], axis=0)
    self.mu_1 = np.mean(x[y == 1], axis=0)
    self.sigma = np.cov(x.T)
    theta = np.transpose(self.mu_1 - self.mu_0) @ np.linalg.inv(self.sigma)
    theta_0 = np.log(
        self.phi /
        (1 -
         self.phi)) - 0.5 * np.transpose(self.mu_1 - self.mu_0) @ np.linalg.inv(
             self.sigma) @ (self.mu_1 + self.mu_0)
    self.theta = np.append(theta_0, theta)

  def predict(self, x):
    """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
    # We calculate p(y=1|x) = p(x|y=1)p(y=1) / p(x)
    # select the class with the highest probability
    # we can ignore the denominator since it is the same for both classes
    p_1 = self.phi * np.exp(-0.5 * np.sum(
        (x - self.mu_1) @ np.linalg.inv(self.sigma) * (x - self.mu_1), axis=1))
    p_0 = (1 - self.phi) * np.exp(-0.5 * np.sum(
        (x - self.mu_0) @ np.linalg.inv(self.sigma) * (x - self.mu_0), axis=1))
    return np.where(p_1 > p_0, 1, 0)


if __name__ == '__main__':
  main(train_path='ds1_train.csv',
       valid_path='ds1_valid.csv',
       plot_path='gda_pred_1.png',
       save_path='gda_pred_1.txt')

  main(train_path='ds2_train.csv',
       valid_path='ds2_valid.csv',
       plot_path='gda_pred_2.png',
       save_path='gda_pred_2.txt')
