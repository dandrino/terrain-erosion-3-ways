#!/usr/bin/python3

# Generates a PNG containing a hillshaded version of the terrain height.

import os
import sys
import argparse
import util


def main(argv):
  parser = argparse.ArgumentParser(description="Generates a PNG containing a hillshaded version of the terrain height.")
  parser.add_argument("input_array", help="<input_array.np[yz]> (include file extension)")
  parser.add_argument("output_image", help="<output_image.png> (include file extension)")
  args = parser.parse_args()

  my_dir = os.path.dirname(argv[0])
  output_dir = os.path.join(my_dir, 'output')
  
  input_path = args.input_array
  output_path = os.path.join(output_dir, args.output_image)

  height, land_mask = util.load_from_file(input_path)
  util.save_as_png(util.hillshaded(height, land_mask=land_mask), output_path)


if __name__ == '__main__':
  main(sys.argv)
