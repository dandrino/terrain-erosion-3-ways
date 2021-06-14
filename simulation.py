#!/usr/bin/python3

# Semi-phisically-based hydraulic erosion simulation. Code is inspired by the 
# code found here:
#   http://ranmantaru.com/blog/2011/10/08/water-erosion-on-heightmap-terrain/
# With some theoretical inspiration from here:
#   https://hal.inria.fr/inria-00402079/document

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import argparse
import util


# Smooths out slopes of `terrain` that are too steep. Rough approximation of the
# phenomenon described here: https://en.wikipedia.org/wiki/Angle_of_repose
def apply_slippage(terrain, repose_slope, cell_width):
  delta = util.simple_gradient(terrain) / cell_width
  smoothed = util.gaussian_blur(terrain, sigma=1.5)
  should_smooth = np.abs(delta) > repose_slope
  result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)
  return result


def main(argv):
  parser = argparse.ArgumentParser(description="Run a terrain erosion simulation.")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("-f", "--file", help="Run simulation using a grayscale input image file instead of generating a new fBm noise. (Only works with a square image.) If not specified then noise will be generated.")
  group.add_argument("-s", "--seed", type=int, help="Noise generator seed. If not specified then a random seed will be used. SEED MUST BE AN INTEGER.")
  parser.add_argument("-o", "--output", help="Output simulation file name (without file extension). If not specified then the default file name will be used.")
  parser.add_argument("--snapshot", action="store_true", help="Save a numbered image of every iteration.")
  parser.add_argument("--png", action="store_true", help="Automatically save a png of the simulation.")
  args = parser.parse_args()

  my_dir = os.path.dirname(argv[0])
  output_dir = os.path.join(my_dir, 'output')
  try: os.mkdir(output_dir)
  except: pass

  if args.output:
    output_path = os.path.join(output_dir, args.output)
  else:
    output_path = os.path.join(output_dir, 'simulation')

  if args.seed:
    input_seed = args.seed
  else:
    input_seed = None

  # Grid dimension constants if using fBm noise
  full_width = 200
  dim = 512
  shape = [dim] * 2
  cell_width = full_width / dim
  cell_area = cell_width ** 2

  # `terrain` represents the actual terrain height we're interested in
  if not args.file:
    terrain = util.fbm(shape, -2.0, seed=input_seed)
  else:
    terrain = util.image_to_array(args.file)

    dim = terrain.shape[0]
    shape = terrain.shape
    cell_width = full_width / dim
    cell_area = cell_width ** 2

  # Snapshotting parameters. Only needed for generating the simulation
  # timelapse.
  if args.snapshot:
    snapshot_dir = os.path.join(output_dir, 'sim_snaps')
    snapshot_file_template = 'sim-%05d.png'
    try: os.mkdir(snapshot_dir)
    except: pass

  # Water-related constants
  rain_rate = 0.0008 * cell_area
  evaporation_rate = 0.0005

  # Slope constants
  min_height_delta = 0.05
  repose_slope = 0.03
  gravity = 30.0
  gradient_sigma = 0.5

  # Sediment constants
  sediment_capacity_constant = 50.0
  dissolving_rate = 0.25
  deposition_rate = 0.001

  # The numer of iterations is proportional to the grid dimension. This is to 
  # allow changes on one side of the grid to affect the other side.
  iterations = int(1.4 * dim)

  # `sediment` is the amount of suspended "dirt" in the water. Terrain will be
  # transfered to/from sediment depending on a number of different factors.
  sediment = np.zeros_like(terrain)

  # The amount of water. Responsible for carrying sediment.
  water = np.zeros_like(terrain)

  # The water velocity.
  velocity = np.zeros_like(terrain)

  # Optionally save the unmodified starting noise if we're not using file input.
  if args.snapshot and args.png and not args.file:
    fbm_path = output_path + '_fbm.png'
    util.save_as_png(terrain, fbm_path)

  for i in range(0, iterations):
    print('Iteration: %d / %d' % (i + 1, iterations))

    # Set a deterministic seed for our random number generator
    rng = np.random.default_rng(i)

    # Add precipitation. This is done via simple uniform random distribution,
    # although other models use a raindrop model
    water += rng.random(shape) * rain_rate

    # Use a different RNG seed for the next step
    rng = np.random.default_rng(i + 3)

    # Compute the normalized gradient of the terrain height to determine where 
    # water and sediment will be moving.
    gradient = np.zeros_like(terrain, dtype='complex')
    gradient = util.simple_gradient(terrain)
    gradient = np.select([np.abs(gradient) < 1e-10],
                             [np.exp(2j * np.pi * rng.random(shape))],
                             gradient)
    gradient /= np.abs(gradient)

    # Compute the difference between the current height the height offset by
    # `gradient`.
    neighbor_height = util.sample(terrain, -gradient)
    height_delta = terrain - neighbor_height
    
    # The sediment capacity represents how much sediment can be suspended in
    # water. If the sediment exceeds the quantity, then it is deposited,
    # otherwise terrain is eroded.
    sediment_capacity = (
        (np.maximum(height_delta, min_height_delta) / cell_width) * velocity *
        water * sediment_capacity_constant)
    deposited_sediment = np.select(
        [
          height_delta < 0, 
          sediment > sediment_capacity,
        ], [
          np.minimum(height_delta, sediment),
          deposition_rate * (sediment - sediment_capacity),
        ],
        # If sediment <= sediment_capacity
        dissolving_rate * (sediment - sediment_capacity))

    # Don't erode more sediment than the current terrain height.
    deposited_sediment = np.maximum(-height_delta, deposited_sediment)

    # Update terrain and sediment quantities.
    sediment -= deposited_sediment
    terrain += deposited_sediment
    sediment = util.displace(sediment, gradient)
    water = util.displace(water, gradient)

    # Smooth out steep slopes.
    terrain = apply_slippage(terrain, repose_slope, cell_width)

    # Update velocity
    velocity = gravity * height_delta / cell_width
  
    # Apply evaporation
    water *= 1 - evaporation_rate

    # Snapshot, if applicable.
    if args.snapshot:
      snapshot_path = os.path.join(snapshot_dir, snapshot_file_template % i)
      util.save_as_png(terrain, snapshot_path)

  # Normalize terrain values before saving.
  result = util.normalize(terrain)

  np.save(output_path, result)
  # Optionally save out an image as well.
  if args.png:
    util.save_as_png(result, output_path + '_gray.png')
    util.save_as_png(util.hillshaded(result), output_path + '_hill.png')

  
if __name__ == '__main__':
  main(sys.argv)
