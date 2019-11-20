# Linear Regression

## Normalization and setup

Load a dataset containing age, weight and height.

- Age (dimension) is in years
- Weight (dimension)  is in kg
- Height (classifier) is in meters

Use standard deviation to scale and normalize the data.

## Use Gradient Descent

Use gradient descent to find the regression model.

Use the following learning rates:
learning rates: α ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}
iterations: 100

Compare convergence rate when α is small vs. large. Then determine your own learning rate based on what you've confirmed so far. Then, using this new learning rate, run the algorithm for your own choice of number of iterations.

# Run the program

Run the program with:
```
python3 problem2.py input2.csv output2.csv
```

# Output

The output must contain the above learning rates plus your own learning rate, creating 10 cases in total. Output should contain the following per case/line: alpha, number_of_iterations, b_0, b_age, and b_weight

# Visualization

To visualize the result of each case of gradient descent, you can use matplotlib to output an image for each linear regression model in three-dimensional space. For instance, you can plot each dimension on the xy-plane, and plot the regression equation as a plane in xyz-space. An example is shown above for reference.

