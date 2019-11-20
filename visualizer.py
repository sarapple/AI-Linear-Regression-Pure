from mpl_toolkits import mplot3d

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

matplotlib.get_cachedir()

class Visualizer:
  @staticmethod
  def draw_scatter(input_values, weights, axes):
    x_points = [input_value[0] for input_value in input_values]
    y_points = [input_value[1] for input_value in input_values]
    z_points = [(
      weights[0]
      + (weights[1] * input_value[0])
      + (weights[2] * input_value[1]
    )) for input_value in input_values]
    
    axes.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')

  @staticmethod
  def draw_wireframe(input_values, weights, axes):
    def z_function(x_point, y_point):
      return (
        weights[0]
        + (weights[1] * x_point)
        + (weights[2] * y_point)
      )
    x_points = [input_value[0] for input_value in input_values]
    y_points = [input_value[1] for input_value in input_values]

    X, Y = np.meshgrid(x_points, y_points)
    Z = z_function(X, Y)
    axes.plot_wireframe(X, Y, Z, color='green', alpha = 0.2)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

  @staticmethod
  def draw_chart(input_values, weights, file_name):
    axes = plt.axes(projection="3d")

    Visualizer.draw_wireframe(input_values, weights, axes)
    Visualizer.draw_scatter(input_values, weights, axes)

    plt.savefig(file_name + ".png")
    plt.clf()
