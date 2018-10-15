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
import util


# Smooths out slopes of `terrain` that are too steep. Rough approximation of the
# phenomenon described here: https://en.wikipedia.org/wiki/Angle_of_repose
def apply_slippage(terrain, repose_slope, cell_width):
  # A simple cross-shaped kernel.
  kernel = np.zeros_like(terrain)
  kernel[0][0] = 1.0
  kernel[1][0] = 1.0
  kernel[-1][0] = 1.0
  kernel[0][1] = 1.0
  kernel[0][-1] = 1.0
  kernel /= kernel.sum()

  delta = util.gaussian_gradient(terrain, sigma=0.5) / cell_width
  smoothed = np.fft.ifft2(np.fft.fft2(terrain) * np.fft.fft2(kernel)).real

  should_smooth = np.abs(delta) > repose_slope
  result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)

  return result


def main(argv):
  # Grid dimension constants
  full_width = 200
  dim = 512
  shape = [dim] * 2
  cell_width = full_width / dim
  cell_area = cell_width ** 2

  # Snapshotting parameters. Only needed for generating the simulation
  # timelapse.
  enable_snapshotting = False
  my_dir = os.path.dirname(argv[0])
  snapshot_dir = os.path.join(my_dir, 'sim_snaps')
  snapshot_file_template = 'sim-%05d.png'
  if enable_snapshotting:
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
  sediment_capacity_constant = 10.0
  deposition_rate = 0.005
  dissolving_rate = 0.35

  # The numer of iterations is proportional to the grid dimension. This is to 
  # allow changes on one side of the grid to affect the other side.
  iterations = int(2.5 * dim)

  # `terrain` represents the actual terrain height we're interested in
  terrain = util.fbm(shape, -2.0)

  # `sediment` is the amount of suspended "dirt" in the water. Terrain will be
  # transfered to/from sediment depending on a number of different factors.
  sediment = np.zeros_like(terrain)

  # The amount of water. Responsible for carrying sediment.
  water = np.zeros_like(terrain)

  # The water velocity.
  velocity = np.zeros_like(terrain)

  for i in range(0, iterations):
    print('%d / %d' % (i + 1, iterations))

    # Add precipitation. This is done by via simple uniform random distribution,
    # although other models use a raindrop model
    water += np.random.rand(*shape) * rain_rate

    # Compute the normalized gradient of the terrain height to determine where 
    # water and sediment will be moving.
    gradient = np.zeros_like(terrain, dtype='complex')
    gradient = util.gaussian_gradient(terrain, sigma=gradient_sigma)
    gradient = np.select([np.abs(gradient) < 1e-10],
                         [np.exp(2j * np.pi * np.random.rand(*shape))],
                         gradient)
    gradient /= np.abs(gradient)

    # Compute the difference between teh current height the height offset by
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
    if enable_snapshotting:
      output_path = os.path.join(snapshot_dir, snapshot_file_template % i)
      util.save_as_png(terrain, output_path)


  np.save('simulation', util.normalize(terrain))

  
if __name__ == '__main__':
  main(sys.argv)
