import sys
import numpy as np

from gradient_descent import GradientDescent
from normalizer import Normalizer
from visualizer import Visualizer
from reporter import Reporter
from reader import Reader

def main():
  input_csv_file_name = sys.argv[1]
  output_csv_file_name = sys.argv[2]

  # input values are in the form of [age, weight, height]
  input_values = Reader.csv(input_csv_file_name)

  cases = [
    (0.001, 100),
    (0.005, 100),
    (0.01, 100),
    (0.05, 100),
    (0.1, 100),
    (0.5, 100),
    (1, 100),
    (5, 100),
    (10, 100),
    #custom
    (0.01, 1000), 
  ]

  Reporter.write_output(
    file_name = output_csv_file_name,
    content = "",
    should_overwrite_file = True
  )

  for learning_rate, iterations in cases:
    # reset betas each time
    betas = [
      # intercept 0
      0,
      # age
      0,
      # weight
      0
    ]

    training_inputs = [[x[0], x[1]] for x in input_values]
    expected_classifiers = [x[2] for x in input_values]

    betas = GradientDescent.run(
      training_inputs = training_inputs,
      expected_classifiers = expected_classifiers,
      learning_rate = learning_rate,
      betas = betas,
      iterations = iterations
    )

    # write lines to output file
    Reporter.write_output(
      file_name = output_csv_file_name,
      content = ",".join(map(str, [learning_rate, iterations, betas[0], betas[1], betas[2]])) + "\n",
    )

    Visualizer.draw_chart(
      input_values = training_inputs,
      weights = betas,
      file_name = "figures/figure_" + "_".join(map(str, [learning_rate, iterations]))
    )

if __name__ == '__main__':
    main()
