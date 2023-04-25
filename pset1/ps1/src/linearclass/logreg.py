import numpy as np
import util


def main(train_path, valid_path, plot_path, save_path):
  """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
  x_train, y_train = util.load_dataset(csv_path=train_path, add_intercept=True)
  x_valid, y_valid = util.load_dataset(csv_path=valid_path, add_intercept=True)

  # Train a logistic regression classifier
  # Plot decision boundary on top of validation set set
  # Use np.savetxt to save predictions on eval set to save_path
  classifier = LogisticRegression()
  classifier.fit(x_train, y_train)
  print(f"theta: {classifier.theta}")
  util.plot(x_valid, y_valid, classifier.theta, plot_path)


class LogisticRegression:
  """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

  def __init__(self,
               step_size=0.01,
               max_iter=1000000,
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

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def h(self, x, theta):
    return self.sigmoid(np.dot(x, theta))

  # np.log is the natural log (base e)
  # It is performed element-wise
  # Similarly, np.exp is the natural exponential
  # It is also performed element-wise
  # "*" is element-wise multiplication in numpy arrays.
  # They are not matrix multiplication. They should have the same shape.
  # (1 - y) is also element-wise subtraction
  # + is element-wise addition
  def J(self, x, y):
    h = self.h(x, self.theta)
    total_value = y * np.log(h) + (1 - y) * np.log(1 - h)
    return -np.sum(total_value) / len(y)

  def J_prime(self, x: np.ndarray, y: np.ndarray):
    h = self.h(x, self.theta)
    # Very very interesting.
    return np.dot(x.T, h - y) / len(y)

  def J_hessian(self, x: np.ndarray, y: np.ndarray):
    h = self.h(x, self.theta)
    return np.dot(x.T, np.diag(h * (1 - h)) @ x) / len(y)

  # This x is the x_train
  # This y is the y_train
  # X = (n_examples, dim)
  # Y = (n_examples, 1)
  def fit(self, x: np.ndarray, y: np.ndarray):
    """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
    print(x.shape)
    print(x[0])
    if self.theta is None:
      self.theta = np.zeros(x.shape[1])
    # Now lets do Newton's method
    # We need to find the theta that minimizes J
    iterations = 0
    first_cost = self.J(x, y)
    while iterations < self.max_iter:
      descent = np.linalg.inv(self.J_hessian(x, y)) @ self.J_prime(x, y)
      prev_theta = np.copy(self.theta)
      self.theta -= descent
      # If the difference between the previous theta and the current theta is less than eps, we have converged
      print(
          "Iteration: ",
          iterations,
          "Norm: ",
          np.linalg.norm(self.theta - prev_theta),
      )
      if np.linalg.norm(self.theta - prev_theta) < self.eps:
        break
      iterations += 1

    print("Converged in {} iterations. Down from {} to {}".format(
        iterations, first_cost, self.J(x, y)))

  def predict(self, x):
    """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
    return self.h(x, self.theta)


if __name__ == "__main__":
  main(
      train_path="ds1_train.csv",
      valid_path="ds1_valid.csv",
      plot_path="logreg_pred_1.png",
      save_path="logreg_pred_1.txt",
  )

  main(
      train_path="ds2_train.csv",
      valid_path="ds2_valid.csv",
      plot_path="logreg_pred_2.png",
      save_path="logreg_pred_2.txt",
  )
