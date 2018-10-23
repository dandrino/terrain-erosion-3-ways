#!/usr/bin/python3

# A simple domain warping example.

import numpy as np
import sys
import util


def main(argv):
  shape = (512,) * 2

  values = util.fbm(shape, -2, lower=2.0)
  offsets = 150 * (util.fbm(shape, -2, lower=1.5) +
                   1j * util.fbm(shape, -2, lower=1.5))
  result = util.sample(values, offsets)
  np.save('domain_warping', result)


if __name__ == '__main__':
  main(sys.argv)
