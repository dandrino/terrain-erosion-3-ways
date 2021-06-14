#!/usr/bin/python3

# A demo of ridge noise.

import numpy as np
import os
import sys
import argparse
import util


def noise_octave(shape, f):
  return util.fbm(shape, -1, lower=f, upper=(2 * f))

def main(argv):
  parser = argparse.ArgumentParser(description="Generate ridge-like fBm noise.")
  parser.add_argument("-o", "--output", help="Output file name (without file extension). If not specified then the default file name will be used.")
  parser.add_argument("--png", action="store_true", help="Automatically save a png of the noise.")
  args = parser.parse_args()

  my_dir = os.path.dirname(argv[0])
  output_dir = os.path.join(my_dir, 'output')

  if args.output:
    output_path = os.path.join(output_dir, args.output)
  else:
    output_path = os.path.join(output_dir, 'ridge')

  shape = (512,) * 2

  values = np.zeros(shape)
  for p in range(1, 10):
    a = 2 ** p
    values += np.abs(noise_octave(shape, a) - 0.5)/ a 
  result = (1.0 - util.normalize(values)) ** 2

  np.save(output_path, result)

  # Optionally save out an image as well.
  if args.png:
    util.save_as_png(result, output_path + '_gray.png')
    util.save_as_png(util.hillshaded(result), output_path + '_hill.png')


if __name__ == '__main__':
  main(sys.argv)
