import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import utils.jsonreader as jsonreader


def get_colmap_labels(folder, file='/sparse/0/images.txt'):
  """
  Saves relative translations and rotations to labels.json
  :param f: Current folder
  :param file: File to retrieve poses from
  :return:
  """
  file = folder + file
  if not os.path.exists(file):
    print("File " + file + " doesn't exist.")
    return
  lines = []
  with open(file, 'r+') as fi:
    # First few header lines of images.txt
    for i in range(4):
      fi.readline()
    while True:
      line = fi.readline()
      if not line:
        break
      if "png" in line:
        lines.append(line)

  lines.sort(key=lambda x: x.split(" ")[-1].strip())
  num_imgs = len(os.listdir(folder+"/images"))

  if len(lines) < num_imgs:
      print("bad colmap for " + folder)
      print(len(lines))
      print(num_imgs)

  print("Colmapping for", folder, "with ", len(lines), "lines.")
  dct = {}
  for l in range(len(lines) - 1):
    curr = lines[l]
    next = lines[l + 1]
    cid, cQW, cQX, cQY, cQZ, cTX, cTY, cTZ, _, cname = curr.split()
    nid, nQW, nQX, nQY, nQZ, nTX, nTY, nTZ, _, nname = next.split()

    t = [float(cTX) - float(nTX), float(cTY) - float(nTY), float(cTZ) - float(nTZ)]
    q1 = np.array([cQW, cQX, cQY, cQZ])
    q2 = np.array([nQW, nQX, nQY, nQZ])

    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)

    r = rot1.inv() * rot2
    r = r.as_dcm()
    r = r.tolist()

    dct[cname] = (t, r)

  with open(folder+'/labels.json', 'w') as fp:
    json.dump(dct, fp, indent=4, sort_keys=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--folder', type=int, default=0)
    parser.add_argument('--type', choices=["opensfm", "colmap"], default="colmap")
    parser.add_argument('--txt', default='/sparse/0/images.txt')


    args = parser.parse_args()
    dir = args.dir

    # convert to dictionary
    params = vars(args)

    if params['type'] == None:
      raise Exception("Opensfm or colmap?")

    if args.type == "colmap":
      get_labels = get_colmap_labels

    if params['folder'] == 0:
      get_labels(dir, args.txt)

    else:
      folders = sorted(os.listdir(dir))

      for f in folders:
        if not f.startswith("."):
          get_labels(dir + "/" + f)


if __name__ == "__main__":
  main()
