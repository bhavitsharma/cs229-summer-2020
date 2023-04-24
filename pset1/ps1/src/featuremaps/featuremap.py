import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0


class LinearModel(object):
  """Base class for linear models."""

  def __init__(self, theta=None):
    """
        Args:
            theta: Weights vector for the model.
        """
    self.theta = theta

  def fit(self, X, y):
    """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
    # Use the normal equations to find the optimal theta
    # The equiation is theta = (X^T X)^-1 X^T y
    # Since the inverse don't really exist sometimes
    # we need to np.linalg.solve to solve the equation
    self.theta = np.linalg.solve(X.T @ X, y @ X)

  def create_poly(self, k, X):
    """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
    new_X = np.ones((X.shape[0], k + 1))
    for i in range(1, k + 1):
      new_X[:, i] = np.power(X[:, 1], i)
    return new_X

  def create_sin(self, k, X):
    """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
    new_X = np.ones((X.shape[0], k + 2))
    for i in range(1, k + 1):
      new_X[:, i] = np.power(X[:, 1], i)
    new_X[:, k + 1] = np.sin(X[:, 1])
    return new_X

  def predict(self, X):
    """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
    return self.theta @ X.T


def run_exp(train_path,
            sine=False,
            ks=[1, 2, 3, 5, 10, 20],
            filename='plot.png'):
  train_x, train_y = util.load_dataset(train_path, add_intercept=True)
  plot_x = np.ones([1000, 2])
  plot_x[:, 1] = np.linspace(-factor * np.pi, factor * np.pi, 1000)
  plt.figure()
  plt.scatter(train_x[:, 1], train_y)

  mapped_train_x = None
  mapped_plot_x = None

  for k in ks:
    '''
        Our objective is to train models and perform predictions on plot_x data
        '''
    model = LinearModel()
    if sine:
      mapped_train_x = model.create_sin(k, train_x)
      mapped_plot_x = model.create_sin(k, plot_x)
    else:
      mapped_train_x = model.create_poly(k, train_x)
      mapped_plot_x = model.create_poly(k, plot_x)

    model.fit(mapped_train_x, train_y)
    y_pred = model.predict(mapped_plot_x)

    plt.ylim(-2, 2)
    plt.plot(plot_x[:, 1], y_pred, label='k=%d' % k)

  plt.legend()
  plt.savefig(filename)
  plt.clf()


def main(train_path, small_path, eval_path):
  '''
    Run all expetriments
    '''
  run_exp(train_path, sine=False, ks=[3], filename='b.png')
  run_exp(train_path, sine=False, ks=[3, 5, 10, 20], filename='c.png')
  run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename='d.png')
  run_exp(small_path, ks=[1, 2, 5, 10, 20], filename='e.png')


if __name__ == '__main__':
  main(train_path='train.csv', small_path='small.csv', eval_path='test.csv')
