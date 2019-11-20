import math

from functools import reduce

class Normalizer:
  @staticmethod
  def get_stdev(inputs):
    mean = Normalizer.get_mean(inputs)
    num_items = len(inputs)
    sum_of_squared_diff = reduce(lambda x, y: x + y, [((x - mean) ** 2) for x in inputs])
    
    return math.sqrt(sum_of_squared_diff / num_items)

  @staticmethod
  def get_mean(inputs):
    total = reduce(lambda x, y: x + y,inputs)
    num_items = len(inputs)
    mean = total/num_items

    return mean

  @staticmethod
  def normalize_to_stdev(inputs):
    mean = Normalizer.get_mean(inputs)
    stdev = Normalizer.get_stdev(inputs)

    return [((x - mean) / stdev) for x in inputs]
