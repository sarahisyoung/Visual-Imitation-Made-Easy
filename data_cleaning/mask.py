import glob
import os
from PIL import Image, ImageDraw


def blackout(f):
  print("Masking for: " + f)
  imgs = sorted(glob.glob(f + "/images/*"))
  for i in imgs:
    title = i.split("/")[-1]
    shape = [(750, 630), (1250, 1500)]
    # Creating rectangle
    im = Image.open(i)
    img1 = ImageDraw.Draw(im)
    img1.rectangle(shape, fill="black", outline="black")

    if not os.path.exists(f + "/masked_images/"):
      os.mkdir(f + "/masked_images/")
    print(f + "/masked_images/" + title + ".png")
    im.save(f + "/masked_images/" + title + ".png")



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir')
parser.add_argument('--folder', type=int, default=1)

args = parser.parse_args()
dir = args.dir

if args.folder:
  runs = sorted(glob.glob(dir + "/*"))
  for f in runs:
    blackout(f)
else:
  blackout(dir)
