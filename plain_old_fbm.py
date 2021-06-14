#!/usr/bin/python3

# A demo of just regular FBM noise

import numpy as np
import os
import sys
import argparse
import util


def main(argv):
  parser = argparse.ArgumentParser(description="Generate fractional Brownian motion (fBm) noise.")
  parser.add_argument("-s", "--seed", type=int, help="Noise generator seed. If not specified then a random seed will be used. SEED MUST BE AN INTEGER.")
  parser.add_argument("-o", "--output", help="Output noise file name (without file extension). If not specified then the default file name will be used.")
  parser.add_argument("--png", action="store_true", help="Automatically save a png of the noise.")
  args = parser.parse_args()

  my_dir = os.path.dirname(argv[0])
  output_dir = os.path.join(my_dir, 'output')

  if args.output:
    output_path = os.path.join(output_dir, args.output)
  else:
    output_path = os.path.join(output_dir, 'plain_fbm')

  shape = (512,) * 2
  if args.seed:
    input_seed = args.seed
  else:
    input_seed = None

  # Generate the noise.
  fbm_noise = util.fbm(shape, -2, lower=2.0, seed=input_seed)
  np.save(output_path, fbm_noise)

  # Optionally save out an image as well.
  if args.png:
    util.save_as_png(fbm_noise, output_path + '_gray.png')
    util.save_as_png(util.hillshaded(fbm_noise), output_path + '_hill.png')


if __name__ == '__main__':
  main(sys.argv)
