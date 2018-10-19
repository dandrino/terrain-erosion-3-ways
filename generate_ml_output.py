#!/usrb/bin/python3

# Prouces terrain samples from the trained generator network.

import numpy as np
import os
import pickle
import sys
import tensorflow as tf


def main(argv):
  # First argument is the path to the progressive_growing_of_gans clone. This
  # is needed to for proper loading of the weights via pickle.
  # Second argument is the network weights pickle file.
  # Third argument is the number of output samples to generate. Defaults to 20
  if len(argv) < 3:
    print('Usage: %s path/to/progressive_growing_of_gans weights.pkl '
           '[number_of_samples]' % argv[0])
    sys.exit(-1)
  my_dir = os.path.dirname(argv[0])
  pgog_path = argv[1]
  weight_path = argv[2]
  num_samples = int(argv[3]) if len(argv) >= 4 else 20

  # Load the GAN tensors.
  tf.InteractiveSession()
  sys.path.append(pgog_path)
  with open(weight_path, 'rb') as f:
    G, D, Gs = pickle.load(f)

  # Generate input vectors.
  latents = np.random.randn(num_samples, *Gs.input_shapes[0][1:])
  labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

  # Run generator to create samples.
  samples = Gs.run(latents, labels)

  # Make output directory
  output_dir = os.path.join(my_dir, 'ml_outputs')
  try: os.mkdir(output_dir)
  except: pass

  # Write outputs.
  for idx in range(samples.shape[0]):
    sample = (np.clip(np.squeeze((samples[idx, 0, :, :] + 1.0) / 2), 0.0, 1.0)
                 .astype('float64'))
    np.save(os.path.join(output_dir, '%d.npy' % idx), sample)


if __name__ == '__main__':
  main(sys.argv)
