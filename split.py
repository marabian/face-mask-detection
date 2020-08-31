import os
import sys
import random
import csv
import shutil
from PIL import Image

# read command line arguments
# create directories
# read csv and shuffle list
# crop images and save to train, valid, test based on specfied ratio

usage_msg = "Usage:\n  python3 split.py <path to images> <path to csv file> <train %> <valid %> <test %>"

if len(sys.argv) < 5:
  print(usage_msg)
  sys.exit()

img_path = sys.argv[1]
csv_path = sys.argv[2]
ptrain = int(sys.argv[3])
pvalid = int(sys.argv[4])
ptest = int(sys.argv[5])

if (ptrain + pvalid + ptest) != 100:
  print(usage_msg + "\n<train %>, <valid %>, <test %> must add up to 100")
  sys.exit()

# create directories
dir_name = "data"
if os.path.isdir(dir_name) is False:
  os.makedirs(dir_name)
  os.makedirs(dir_name + '/train/mask')
  os.makedirs(dir_name + '/train/no-mask')
  os.makedirs(dir_name + '/valid/mask')
  os.makedirs(dir_name + '/valid/no-mask')
  os.makedirs(dir_name + '/test/mask')
  os.makedirs(dir_name + '/test/no-mask')
else:
  sys.exit()

# read csv file
with open(csv_path) as fp:
  reader = csv.reader(fp, delimiter=",", quotechar='"')

  fields = next(reader, None)
  indexName = fields.index('name')
  indexClassname = fields.index('classname')

  # crop and save
  def cropnsave(row, to):
    ext = row[indexName].split(".")[1]
    im = Image.open(os.path.join(img_path, row[indexName]))
    (x1, x2, y1, y2) = (int(row[1]),int(row[2]),int(row[3]),int(row[4]))
    new_im = im.crop((x1, x2, y1, y2)) # crop
    new_im.save(os.path.join(dir_name + to, str(i) + "." + ext))
    print(row[indexName], end="\r", flush=True)

  print("cropping images")
  i = 0
  for row in reader:
    # MASK
    if row[indexClassname] == "face_with_mask":
      cropnsave(row, "/train/mask")
    # NO MASK
    elif row[indexClassname] == "face_no_mask":
      cropnsave(row, "/train/no-mask")

    i += 1

  print(i)

# move images
print("moving images")

def move(loc, dest, files):
  for file in files:
    shutil.move(os.path.join(loc, file), os.path.join(dest, file))
    print(file, end="\r", flush=True)

cwd = os.getcwd()

# mask
path = os.path.join(cwd, "data/train/mask")
dir_list = os.listdir(path)
n = len(dir_list)
random.shuffle(dir_list)

nvalid = int((pvalid / 100) * n)
new_path = os.path.join(cwd, "data/valid/mask")
move(path, new_path, dir_list[0:nvalid])

ntest = int((ptest / 100) * n)
new_path = os.path.join(cwd, "data/test/mask")
move(path, new_path, dir_list[nvalid:(nvalid+ntest)])

# no mask
path = os.path.join(cwd, "data/train/no-mask")
dir_list = os.listdir(path)
n = len(dir_list)
random.shuffle(dir_list)

nvalid = int((pvalid / 100) * n)
new_path = os.path.join(cwd, "data/valid/no-mask")
move(path, new_path, dir_list[0:nvalid])

ntest = int((ptest / 100) * n)
new_path = os.path.join(cwd, "data/test/no-mask")
move(path, new_path, dir_list[nvalid:(nvalid+ntest)])