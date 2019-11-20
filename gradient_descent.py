import numpy as np

from normalizer import Normalizer
from functools import reduce

class GradientDescent:
  @staticmethod
  def normal_equation(X, y):
    X_dot_product = X.dot(X.transpose())
    X_dot_product_to_minus_one = np.linalg.matrix_power(X_dot_product, -1)
    X_y_dot_product = X.dot(y)

    # gives a vector of betas solved via b=((X'X)^-1)*X'y
    return X_dot_product_to_minus_one * X_y_dot_product

  @staticmethod
  def run(training_inputs, expected_classifiers, betas = None, learning_rate = 0.01, iterations = 1):
    if (betas == None):
      betas = list(np.zeros(len(expected_classifiers) + 1))

    n = len(expected_classifiers)

    # normalize the input data
    m_training_inputs_normalized = GradientDescent.normalize_training_inputs(training_inputs)

    # Add bias to all training_inputs (1)
    m_training_inputs_with_bias = np.append(np.ones((n, 1)), m_training_inputs_normalized, axis = 1)

    for _ in range(iterations):
      # derive predicted label from the inputs
      predicted_list = [GradientDescent.get_predicted_label([betas[0], betas[1], betas[2]], [x[0], x[1], x[2]]) for x in m_training_inputs_with_bias]

      # derive error for each beta
      total_errors = [GradientDescent.compute_total_error(
        expected_classifiers = expected_classifiers,
        predicted_classifiers = predicted_list,
        dimension_inputs = m_training_inputs_with_bias[:,dimension_index]
      ) for dimension_index in range(len(betas))]
      betas = [beta - (learning_rate * 1/n * total_errors[dimension_index]) for dimension_index, beta in enumerate(betas)]

    return betas

  @staticmethod
  def normalize_training_inputs(training_inputs):
    m_training_inputs_only = np.array(training_inputs)
    m_dimensions = m_training_inputs_only.transpose()
    m_dimensions_normalized = np.array([Normalizer.normalize_to_stdev(dimension_set) for dimension_set in m_dimensions])
    m_training_inputs_normalized = m_dimensions_normalized.transpose()

    return m_training_inputs_normalized

  @staticmethod
  def compute_total_error(expected_classifiers, predicted_classifiers, dimension_inputs):
    n = len(dimension_inputs)
    all_errors = [
      GradientDescent.computed_error(
        expected_classifier = expected_classifiers[example_index],
        predicted_label = predicted_classifiers[example_index],
        input_value = dimension_inputs[example_index]
      ) for example_index in range(n)
    ]
    
    return reduce(
      lambda x, y: x + y,
      all_errors
    )

  @staticmethod
  def get_predicted_label(betas, input_values):
    return betas[0] + (betas[1] * input_values[1]) + (betas[2] * input_values[2])

  @staticmethod
  def computed_error(expected_classifier, predicted_label, input_value):
    return (predicted_label - expected_classifier) * input_value
