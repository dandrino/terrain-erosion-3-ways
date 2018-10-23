#!/usr/bin/python3

# A demo of just regular FBM noise

import numpy as np
import sys
import util


def main(argv):
  shape = (512,) * 2
  np.save('fbm', util.fbm(shape, -2, lower=2.0))


if __name__ == '__main__':
  main(sys.argv)
