import argparse
import glob
import os

import cv2


def makeTrainVideo(folder, save):
  img_array = []
  imgs = sorted(glob.glob(folder + '/trainimages/*'))
  print(folder + ": train")
  if len(imgs) == 0:
    return
  for filename in imgs:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    name = "-".join(folder.strip().split('/')[1:])
  out = cv2.VideoWriter(save + '/train_' + name + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 3, size)

  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()


def makeVideo(folder, save, which="train"):
  img_array = []
  imgs = sorted(glob.glob(folder + '/' + which + 'images/*'))
  print(folder + '/' + which + 'images/*')
  if len(imgs) == 0:
    print("no images")
    return
  for filename in imgs:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

  name = "-".join(folder.strip().split('/')[1:])

  out = cv2.VideoWriter(save + '/' + which + '_' + name + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 3, size)

  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type=str)
  parser.add_argument('--out', type=str)
  parser.add_argument('--folder', type=int, default=0)
  parser.add_argument('--remove', type=int, default=0)

  args = parser.parse_args()

  if args.folder == 1:
    folds = sorted(glob.glob(args.dir + "/trashpolicy*"))
    completed = os.listdir(args.dir)
    for f in folds:
      if args.out == None:
        out = f
        completed = os.listdir(out)
      name = "-".join(f.strip().split('/')[1:])
      nametrain = "train_" + name + ".mp4"
      namev = "val_" + name + ".mp4"
      nametest = "test_" + name + ".mp4"

      if nametrain not in completed or namev not in completed or nametest not in completed:
        if nametrain not in completed:
          makeVideo(f, out, "train")
        if namev not in completed:
          makeVideo(f, out, "val")
        if nametest not in completed:
          makeVideo(f, out, "test")
        completed.append(f[:-4] + ".mp4")
        vids = sorted(glob.glob(out + "/*.avi"))
        for ff in vids:
          os.system("ffmpeg -i " + ff + " " + ff[:-4] + ".mp4")
          os.remove(ff)
      else:
        print("folder done: " + f)
  else:
    out = args.out
    if args.out == None:
      out = args.dir
    folder = args.dir
    makeVideo(folder, out, "test")
    makeVideo(folder, out, "val")
    makeVideo(folder, out, "train")
    vids = sorted(glob.glob(out + "/*.avi"))
    for ff in vids:
      os.system("ffmpeg -i " + ff + " " + ff[:-4] + ".mp4")
      os.remove(ff)


if __name__ == "__main__":
  main()
