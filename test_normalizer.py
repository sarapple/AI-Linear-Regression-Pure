from normalizer import Normalizer

def test_get_stdev():
  inputs = [9, 2, 5, 4, 12, 7, 8, 11, 9, 3, 7, 4, 12, 5, 4, 10, 9, 6, 9, 4]

  result = Normalizer.get_stdev(inputs)

  assert round(result, 3) == 2.983

def test_normalize_to_stdev_1():
  inputs = [900, 200, 500, 400]

  results = [round(x, 2) for x in Normalizer.normalize_to_stdev(inputs)]

  assert results == [1.57, -1.18, 0.0, -0.39]

def test_normalize_to_stdev_2():
  inputs = [9, 2, 5, 4]

  results = [round(x, 2) for x in Normalizer.normalize_to_stdev(inputs)]

  assert results == [1.57, -1.18, 0.0, -0.39]
