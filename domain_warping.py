#!/usr/bin/python3

# A simple domain warping example.

import numpy as np
import os
import sys
import argparse
import util


def main(argv):
  parser = argparse.ArgumentParser(description="Generate domain-warped fBm noise.")
  parser.add_argument("-o", "--output", help="Output file name (without file extension). If not specified then the default file name will be used.")
  parser.add_argument("--png", action="store_true", help="Automatically save a png of the noise.")
  args = parser.parse_args()

  my_dir = os.path.dirname(argv[0])
  output_dir = os.path.join(my_dir, 'output')

  if args.output:
    output_path = os.path.join(output_dir, args.output)
  else:
    output_path = os.path.join(output_dir, 'domain_warping')

  shape = (512,) * 2
  values = util.fbm(shape, -2, lower=2.0)
  offsets = 150 * (util.fbm(shape, -2, lower=1.5) +
                   1j * util.fbm(shape, -2, lower=1.5))
  result = util.sample(values, offsets)
  np.save(output_path, result)

  # Optionally save out an image as well.
  if args.png:
    util.save_as_png(result, output_path + '_gray.png')
    util.save_as_png(util.hillshaded(result), output_path + '_hill.png')


if __name__ == '__main__':
  main(sys.argv)
