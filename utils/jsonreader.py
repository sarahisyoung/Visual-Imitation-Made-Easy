import json
import numpy as np
import os

def get_prediction(f):
  input_file = open(f + '/predicted_translations.json', 'r')
  json_decode = json.load(input_file)
  trans = []
  rot = []
  imgs = []
  for img, vals in json_decode.items():
    t = vals["translation"]
    r = vals["rot_matrix"]

    trans.append(t)

    rot.append(r)
    imgs.append(img)
  rot = np.array(rot)
  trans = np.array(trans)
  imgs = np.array(imgs)

  rel_rot = np.array([np.matmul(rot[i + 1], rot[i].T) for i in range(rot.shape[0] - 1)])



  return imgs, trans, rel_rot

def get_vals(f):
  input_file = open(f + '/labels.json', 'r')
  json_decode = json.load(input_file)
  trans = []
  rot = []
  imgs = []
  items = sorted(json_decode.items(), key=lambda x: x[0])
  for img, vals in items:
    if os.path.exists(f + "/images/" + img):
      t, r = vals
      if max(np.abs(np.array(t))) < 0.09:
        continue
      trans.append(t)
      rot.append(r)
      imgs.append(f + "/images/" + img)
  rot = np.array(rot)
  trans = np.array(trans)
  imgs = np.array(imgs)

  return imgs, trans, rot

def get_gripper(f, existing_imgs):
  if not os.path.exists(f + '/gripper_labels.json'):
    return [], {}
  input_file = open(f + '/gripper_labels.json', 'r')
  json_decode = json.load(input_file)
  status = {}
  imgs = []
  items = sorted(json_decode.items(), key=lambda x: x[0])
  for img, vals in items:
    if os.path.exists(f + "/images/" + img) and f + "/images/" + img in existing_imgs:

      status[f + "/images/" + img] = vals
      imgs.append(f + "/images/" + img)
  imgs = np.array(imgs)

  return imgs, status

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('dir')

    args = parser.parse_args()
