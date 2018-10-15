#!/usr/bin/python3

# Extracts the underlying heightmap in each file in zip_files/ and writes the 
# numpy array to array_files/

import csv
import json
import numpy as np
import os
from osgeo import gdal
import re
import shutil
import sys
import tempfile
import util
import zipfile


# Extracts the IMG file from the ZIP archive and returns a Numpy array, or None # if reading or parsing failed.
def get_img_array_from_zip(zip_file, img_name):
  with tempfile.NamedTemporaryFile() as temp_file:
    # Copy to temp file.
    with zip_file.open(img_name) as img_file:
        shutil.copyfileobj(img_file, temp_file)

    # Extract as numpy array.
    geo = gdal.Open(temp_file.name)
    return geo.ReadAsArray() if geo is not None else None


def main(argv):
  my_dir = os.path.dirname(argv[0])
  input_dir = os.path.join(my_dir, 'zip_files')
  output_dir = os.path.join(my_dir, 'array_files')

  if len(argv) != 2:
    print('Usage: %s <ned_file.csv>' % (argv[0]))
    sys.exit(-1)

  csv_path = argv[1]

  # Make the output directory if it doesn't exist yet.
  try: os.mkdir(output_dir)
  except: pass

  entries = util.read_csv(csv_path)
  for index, entry in enumerate(entries):
    src_id = entry['sourceId']
    print('(%d / %d) Processing %s' % (index + 1, len(entries), src_id))
    zip_path = os.path.join(input_dir, src_id + '.zip')

    try:
      # Go though each zip file.
      with zipfile.ZipFile(zip_path, mode='r') as zf:
        ext_names = [name for name in zf.namelist()
                     if os.path.splitext(name)[1] == '.img']
        # Check if EXT files.
        if len(ext_names) == 0:
          print('No IMG files found for %s' % (src_id))
          continue;

        # Warn if there is more than one IMG file
        if len(ext_names) > 1:
          print('More than one IMG file found for %s: %s' % (src_id, ext_names))

        # Get the bounding box. The string manipulation is required given that 
        # the provided dict is not proper JSON
        bounding_box_raw = entry['boundingBox']
        bounding_box_json = re.sub(r'([a-zA-Z]+):', r'"\1":', bounding_box_raw)
        bounding_box = json.loads(bounding_box_json)

        # Create numpy array from IMG file and write it to output
        array = get_img_array_from_zip(zf, ext_names[0])
        if array is not None:
          output_path = os.path.join(output_dir, src_id + '.npz')
          np.savez(output_path, height=array, **bounding_box)
        else:
          print('Failed to load array for %s' % src_id)
        

    except (zipfile.BadZipfile, IOError) as e:
      # Invalid or missing ZIP file.
      print(e)
      continue
   

if __name__ == '__main__':
  main(sys.argv)
