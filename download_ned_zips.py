#!/usr/bin/python3

# Script to hep in downloading files from USGS. The input file is a CSV
# of 3DEP IMG resources downloaded from https://viewer.nationalmap.gov/basic/
# This script is really a poor man's download manager; use any solution you 
# prefer.

import os
import shutil
import sys
import tempfile
import time
import urllib
import util


# Gets list of src ids of previously downloaded files.
def get_previously_downloaded_ids(dir_path):
  return set((os.path.splitext(file_path)[0]
              for file_path in os.listdir(dir_path)))


# Download the file for `src_id` from `url` to `output_dir`. Uses `tmp_dir` an
# intermediate so that aborted downloads are not included in
# get_previously_downloaded_ids above.
def download_file(src_id, url, output_dir, tmp_dir):
  output_file = src_id + '.zip'
  tmp_path = os.path.join(tmp_dir, output_file)

  # Download and save to temp dir
  try:
    urllib.urlretrieve(url, tmp_path)
  except IOError as e:
    print(e)
    return False

  # Move file in temp dir to final output dir
  shutil.move(tmp_path, output_dir)

  return True


def main(argv):
  my_dir = os.path.dirname(argv[0])
  output_dir = os.path.join(my_dir, 'zip_files')
  tmp_dir = '/tmp'

  if len(argv) != 2:
    print('Usage: %s <ned_file.csv>' % (argv[0]))
    sys.exit(-1)

  csv_path = argv[1]

  try: os.mkdir(output_dir)
  except: pass

  downloaded_ids = get_previously_downloaded_ids(output_dir)
  entries = util.read_csv(csv_path)

  for index, entry in enumerate(entries):
    src_id = entry['sourceId']
    download_url = entry['downloadURL']
    pretty_size = entry['prettyFileSize']
    data_format = entry['format']

    # Don't download the same file more than once.
    if src_id in downloaded_ids:
      print('Skipping %s' % src_id)
      continue

    print('(%d / %d) Processing %s of size %s from %s'
          % (index + 1, len(entries), src_id, pretty_size, download_url))

    # Simple data format sanity check.
    if entry['format'] != 'IMG':
      print('Unknown format %s, ignoring...' % (data_format,))
      continue

    # Download file.
    if not download_file(src_id, download_url, output_dir, tmp_dir):
      print('Failed to download from %s' % download_url)
      continue


if __name__ == '__main__':
  main(sys.argv)
