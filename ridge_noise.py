#!/usr/bin/python3

# A demo of ridge noise.

import numpy as np
import sys
import util

def noise_octave(shape, f):
  return util.fbm(shape, -1, lower=f, upper=(2 * f))

def main(argv):
  dim = 512
  shape = (512,) * 2

  values = np.zeros(shape)
  for p in range(1, 10):
    a = 2 ** p
    values += np.abs(noise_octave(shape, a) - 0.5)/ a 
  result = (1.0 - util.normalize(values)) ** 2

  np.save('ridge', result)


if __name__ == '__main__':
  main(sys.argv)
