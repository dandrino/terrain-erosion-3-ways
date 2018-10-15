#!/usr/bin/python3

# Genreates a PNG containing the terrain height in grayscale.

import util
import sys


def main(argv):
  if len(argv) != 3:
    print('Usage: %s <input_array.np[yz]> <output_image.png>' % (argv[0],))
    sys.exit(-1)

  input_path = argv[1]
  output_path = argv[2]

  height, _ = util.load_from_file(input_path)
  util.save_as_png(height, output_path)


if __name__ == '__main__':
  main(sys.argv)
