#!/usr/bin/python3

import collections
import heapq
import numpy as np
import matplotlib
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import scipy as sp
import scipy.spatial
import skimage.measure
import os
import sys
import argparse
import util


# Returns the index of the smallest value of `a`
def min_index(a): return a.index(min(a))


# Returns an array with a bump centered in the middle of `shape`. `sigma`
# determines how wide the bump is.
def bump(shape, sigma):
  [y, x] = np.meshgrid(*map(np.arange, shape))
  r = np.hypot(x - shape[0] / 2, y - shape[1] / 2)
  c = min(shape) / 2
  return np.tanh(np.maximum(c - r, 0.0) / sigma)


# Returns a list of heights for each point in `points`.
def compute_height(points, neighbors, deltas, get_delta_fn=None):
  if get_delta_fn is None:
    get_delta_fn = lambda src, dst: deltas[dst]

  dim = len(points)
  result = [None] * dim
  seed_idx = min_index([sum(p) for p in points])
  q = [(0.0, seed_idx)]

  while len(q) > 0:
    (height, idx) = heapq.heappop(q)
    if result[idx] is not None: continue
    result[idx] = height
    for n in neighbors[idx]:
      if result[n] is not None: continue
      heapq.heappush(q, (get_delta_fn(idx, n) + height, n))
  return util.normalize(np.array(result))


# Same as above, but computes height taking into account river downcutting.
# `max_delta` determines the maximum difference in neighboring points (to
# give the effect of talus slippage). `river_downcutting_constant` affects how
# deeply rivers cut into terrain (higher means more downcutting).
def compute_final_height(points, neighbors, deltas, volume, upstream,
                         max_delta, river_downcutting_constant):
  dim = len(points)
  result = [None] * dim
  seed_idx = min_index([sum(p) for p in points])
  q = [(0.0, seed_idx)]

  def get_delta(src, dst):
    v = volume[dst] if (dst in upstream[src]) else 0.0
    downcut = 1.0 / (1.0 + v ** river_downcutting_constant) 
    return min(max_delta, deltas[dst] * downcut)

  return compute_height(points, neighbors, deltas, get_delta_fn=get_delta)


# Computes the river network that traverses the terrain.
#   Arguments:
#   * points: The (x,y) coordinates of each point
#   * neghbors: Set of each neighbor index for each point.
#   * heights: The height of each point.
#   * land: Indicates whether each point is on land or water.
#   * directional_interta: indicates how straight the rivers are
#       (0 = no directional inertia, 1 = total directional inertia).
#   * default_water_level: How much water is assigned by default to each point
#   * evaporation_rate: How much water is evaporated as it traverses from along
#       each river edge.
#  
#  Returns a 3-tuple of:
#  * List of indices of all points upstream from each point
#  * List containing the index of the point downstream of each point.
#  * The water volume of each point.
def compute_river_network(points, neighbors, heights, land,
                          directional_inertia, default_water_level,
                          evaporation_rate):
  num_points = len(points)

  # The normalized vector between points i and j
  def unit_delta(i, j):
    delta = points[j] - points[i]
    return delta / np.linalg.norm(delta)

  # Initialize river priority queue with all edges between non-land points to
  # land points. Each entry is a tuple of (priority, (i, j, river direction))
  q = []
  roots = set()
  for i in range(num_points):
    if land[i]: continue
    is_root = True
    for j in neighbors[i]:
      if not land[j]: continue
      is_root = True
      heapq.heappush(q, (-1.0, (i, j, unit_delta(i, j))))
    if is_root: roots.add(i)

  # Compute the map of each node to its downstream node.
  downstream = [None] * num_points

  while len(q) > 0:
    (_, (i, j, direction)) = heapq.heappop(q)

    # Assign i as being downstream of j, assuming such a point doesn't
    # already exist.
    if downstream[j] is not None: continue
    downstream[j] = i

    # Go through each neighbor of upstream point j.
    for k in neighbors[j]:
      # Ignore neighbors that are lower than the current point, or who already
      # have an assigned downstream point.
      if (heights[k] < heights[j] or downstream[k] is not None
          or not land[k]):
        continue

      # Edges that are aligned with the current direction vector are
      # prioritized.
      neighbor_direction = unit_delta(j, k)
      priority = -np.dot(direction, neighbor_direction)

      # Add new edge to queue.
      weighted_direction = util.lerp(neighbor_direction, direction,
                                     directional_inertia)
      heapq.heappush(q, (priority, (j, k, weighted_direction)))


  # Compute the mapping of each node to its upstream nodes.
  upstream = [set() for _ in range(num_points)]
  for i, j in enumerate(downstream):
    if j is not None: upstream[j].add(i)

  # Compute the water volume for each node.
  volume = [None] * num_points
  def compute_volume(i):
    if volume[i] is not None: return
    v = default_water_level
    for j in upstream[i]:
      compute_volume(j)
      v += volume[j]
    volume[i] = v * (1 - evaporation_rate)

  for i in range(0, num_points): compute_volume(i)

  return (upstream, downstream, volume)


# Renders `values` for each triangle in `tri` on an array the size of `shape`.
def render_triangulation(shape, tri, values):
  points = util.make_grid_points(shape)
  triangulation = matplotlib.tri.Triangulation(
      tri.points[:,0], tri.points[:,1], tri.simplices)
  interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)
  return interp(points[:,0], points[:,1]).reshape(shape).filled(0.0)


# Removes any bodies of water completely enclosed by land.
def remove_lakes(mask):
  labels = skimage.measure.label(mask)
  new_mask = np.zeros_like(mask, dtype=bool)
  labels = skimage.measure.label(~mask, connectivity=1)
  new_mask[labels != labels[0, 0]] = True
  return new_mask


def main(argv):
  parser = argparse.ArgumentParser(
    description="Generate terrain from a river network.")
  parser.add_argument("-o", "--output", 
    help="Output file name (without file extension). If not specified then \
    the default file name will be used.")
  parser.add_argument("--png", action="store_true", 
    help="Automatically save a png of the terrain.")
  args = parser.parse_args()

  my_dir = os.path.dirname(argv[0])
  output_dir = os.path.join(my_dir, 'output')
  if args.output:
    output_path = os.path.join(output_dir, args.output)
  else:
    output_path = os.path.join(output_dir, 'river_network')

  dim = 512
  shape = (dim,) * 2
  disc_radius = 1.0
  max_delta = 0.05
  river_downcutting_constant = 1.3
  directional_inertia = 0.4
  default_water_level = 1.0
  evaporation_rate = 0.2

  print ('Generating...')

  print('  ...initial terrain shape')
  land_mask = remove_lakes(
      (util.fbm(shape, -2, lower=2.0) + bump(shape, 0.2 * dim) - 1.1) > 0)
  coastal_dropoff = np.tanh(util.dist_to_mask(land_mask) / 80.0) * land_mask
  mountain_shapes = util.fbm(shape, -2, lower=2.0, upper=np.inf)
  initial_height = ( 
      (util.gaussian_blur(np.maximum(mountain_shapes - 0.40, 0.0), sigma=5.0) 
        + 0.1) * coastal_dropoff)
  deltas = util.normalize(np.abs(util.gaussian_gradient(initial_height))) 

  print('  ...sampling points')
  points = util.poisson_disc_sampling(shape, disc_radius)
  coords = np.floor(points).astype(int)

  print('  ...delaunay triangulation')
  tri = sp.spatial.Delaunay(points)
  (indices, indptr) = tri.vertex_neighbor_vertices
  neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]
  points_land = land_mask[coords[:, 0], coords[:, 1]]
  points_deltas = deltas[coords[:, 0], coords[:, 1]]

  print('  ...initial height map')
  points_height = compute_height(points, neighbors, points_deltas)

  print('  ...river network')
  (upstream, downstream, volume) = compute_river_network(
      points, neighbors, points_height, points_land,
      directional_inertia, default_water_level, evaporation_rate)

  print('  ...final terrain height')
  new_height = compute_final_height(
      points, neighbors, points_deltas, volume, upstream, 
      max_delta, river_downcutting_constant)
  terrain_height = render_triangulation(shape, tri, new_height)
  
  np.savez(output_path, height=terrain_height, land_mask=land_mask)

  # Optionally save out an image as well.
  if args.png:
    util.save_as_png(terrain_height, output_path + '_gray.png')
    util.save_as_png(util.hillshaded(
      terrain_height, land_mask=land_mask), output_path + '_hill.png')


if __name__ == '__main__':
  main(sys.argv)
