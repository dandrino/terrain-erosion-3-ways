# Various common functions.

from PIL import Image
import collections
import csv
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial


# Open CSV file as a dict.
def read_csv(csv_path):
  with open(csv_path, 'r') as csv_file:
    return list(csv.DictReader(csv_file))


# Renormalizes the values of `x` to `bounds`
def normalize(x, bounds=(0, 1)):
  return np.interp(x, (x.min(), x.max()), bounds)


# Fourier-based power law noise with frequency bounds.
def fbm(shape, p, lower=-np.inf, upper=np.inf):
  freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
  freq_radial = np.hypot(*np.meshgrid(*freqs))
  envelope = (np.power(freq_radial, p, where=freq_radial!=0) *
              (freq_radial > lower) * (freq_radial < upper))
  envelope[0][0] = 0.0
  phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
  return normalize(np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope)))


# Returns each value of `a` with coordinates offset by `offset` (via complex 
# values). The values at the new coordiantes are the linear interpolation of
# neighboring values in `a`.
def sample(a, offset):
  shape = np.array(a.shape)
  delta = np.array((offset.real, offset.imag))
  coords = np.array(np.meshgrid(*map(range, shape))) - delta

  lower_coords = np.floor(coords).astype(int)
  upper_coords = lower_coords + 1
  coord_offsets = coords - lower_coords 
  lower_coords %= shape[:, np.newaxis, np.newaxis]
  upper_coords %= shape[:, np.newaxis, np.newaxis]

  result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
                     a[lower_coords[1], upper_coords[0]],
                     coord_offsets[0]),
                lerp(a[upper_coords[1], lower_coords[0]],
                     a[upper_coords[1], upper_coords[0]],
                     coord_offsets[0]),
                coord_offsets[1])
  return result


# Takes each value of `a` and offsets them by `delta`. Treats each grid point
# like a unit square.
def displace(a, delta):
  fns = {
      -1: lambda x: -x,
      0: lambda x: 1 - np.abs(x),
      1: lambda x: x,
  }
  result = np.zeros_like(a)
  for dx in range(-1, 2):
    wx = np.maximum(fns[dx](delta.real), 0.0)
    for dy in range(-1, 2):
      wy = np.maximum(fns[dy](delta.imag), 0.0)
      result += np.roll(np.roll(wx * wy * a, dy, axis=0), dx, axis=1)

  return result


# Returns the gradient of the gaussian blur of `a` encoded as a complex number. 
def gaussian_gradient(a, sigma=1.0):
  [fy, fx] = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in a.shape))
  sigma2 = sigma**2
  g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
  dg = lambda x: g(x) * (x / sigma2)

  fa = np.fft.fft2(a)
  dy = np.fft.ifft2(np.fft.fft2(dg(fy) * g(fx)) * fa).real
  dx = np.fft.ifft2(np.fft.fft2(g(fy) * dg(fx)) * fa).real
  return 1j * dx + dy


# Simple gradient by taking the diff of each cell's horizontal and vertical
# neighbors.
def simple_gradient(a):
  dx = 0.5 * (np.roll(a, 1, axis=0) - np.roll(a, -1, axis=0))
  dy = 0.5 * (np.roll(a, 1, axis=1) - np.roll(a, -1, axis=1))
  return 1j * dx + dy


# Loads the terrain height array (and optionally the land mask from the given 
# file.
def load_from_file(path):
  result = np.load(path)
  if type(result) == np.lib.npyio.NpzFile:
    return (result['height'], result['land_mask'])
  else:
    return (result, None)


# Saves the array as a PNG image. Assumes all input values are [0, 1]
def save_as_png(a, path):
  image = Image.fromarray(np.round(a * 255).astype('uint8'))
  image.save(path)


# Creates a hillshaded RGB array of heightmap `a`.
_TERRAIN_CMAP = LinearSegmentedColormap.from_list('my_terrain', [
    (0.00, (0.15, 0.3, 0.15)),
    (0.25, (0.3, 0.45, 0.3)),
    (0.50, (0.5, 0.5, 0.35)),
    (0.80, (0.4, 0.36, 0.33)),
    (1.00, (1.0, 1.0, 1.0)),
])
def hillshaded(a, land_mask=None, angle=270):
  if land_mask is None: land_mask = np.ones_like(a)
  ls = LightSource(azdeg=angle, altdeg=30)
  land = ls.shade(a, cmap=_TERRAIN_CMAP, vert_exag=10.0,
                  blend_mode='overlay')[:, :, :3]
  water = np.tile((0.25, 0.35, 0.55), a.shape + (1,))
  return lerp(water, land, land_mask[:, :, np.newaxis])


# Linear interpolation of `x` to `y` with respect to `a`
def lerp(x, y, a): return (1.0 - a) * x + a * y


# Returns a list of grid coordinates for every (x, y) position bounded by
# `shape`
def make_grid_points(shape):
  [Y, X] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
  grid_points = np.column_stack([X.flatten(), Y.flatten()])
  return grid_points


# Returns a list of points sampled within the bounds of `shape` and with a
# minimum spacing of `radius`.
# NOTE: This function is fairly slow, given that it is implemented with almost
# no array operations.
def poisson_disc_sampling(shape, radius, retries=16):
  grid = {}
  points = []

  # The bounds of `shape` are divided into a grid of cells, each of which can
  # contain a maximum of one point.
  cell_size = radius / np.sqrt(2)
  cells = np.ceil(np.divide(shape, cell_size)).astype(int)
  offsets = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1),
             (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
  to_cell = lambda p: (p / cell_size).astype('int')

  # Returns true if there is a point within `radius` of `p`.
  def has_neighbors_in_radius(p):
    cell = to_cell(p)
    for offset in offsets:
      cell_neighbor = (cell[0] + offset[0], cell[1] + offset[1])
      if cell_neighbor in grid:
        p2 = grid[cell_neighbor]
        diff = np.subtract(p2, p)
        if np.dot(diff, diff) <= radius * radius:
          return True
    return False      

  # Adds point `p` to the cell grid.
  def add_point(p):
    grid[tuple(to_cell(p))] = p
    q.append(p)
    points.append(p)

  q = collections.deque()
  first = shape * np.random.rand(2)
  add_point(first)
  while len(q) > 0:
    point = q.pop()

    # Make `retries` attemps to find a point within [radius, 2 * radius] from
    # `point`.
    for _ in range(retries):
      diff = 2 * radius * (2 * np.random.rand(2) - 1)
      r2 = np.dot(diff, diff)
      new_point = diff + point
      if (new_point[0] >= 0 and new_point[0] < shape[0] and
          new_point[1] >= 0 and new_point[1] < shape[1] and 
          not has_neighbors_in_radius(new_point) and
          r2 > radius * radius and r2 < 4 * radius * radius):
        add_point(new_point)
  num_points = len(points)

  # Return points list as a numpy array.
  return np.concatenate(points).reshape((num_points, 2))


# Returns an array in which all True values of `mask` contain the distance to
# the nearest False value.
def dist_to_mask(mask):
  border_mask = (np.maximum.reduce([
      np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
      np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
  border_points = np.column_stack(np.where(border_mask > 0))

  kdtree = sp.spatial.cKDTree(border_points)
  grid_points = make_grid_points(mask.shape)

  return kdtree.query(grid_points)[0].reshape(mask.shape)


# Generates worley noise with points separated by `spacing`.
def worley(shape, spacing):
  points = poisson_disc_sampling(shape, spacing)
  coords = np.floor(points).astype(int)
  mask = np.zeros(shape, dtype=bool)
  mask[coords[:, 0], coords[:, 1]] = True
  return normalize(dist_to_mask(mask))


# Peforms a gaussian blur of `a`.
def gaussian_blur(a, sigma=1.0):
  freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
  freq_radial = np.hypot(*np.meshgrid(*freqs))
  sigma2 = sigma**2
  g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
  kernel = g(freq_radial)
  kernel /= kernel.sum()
  return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real
