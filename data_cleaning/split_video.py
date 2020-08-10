import cv2
from os import listdir
import os

def split_video(vid, save, fr):
  """
  Splits video into frames.
  :param vid: Path to video
  :param save: Directory to save frames
  :param fr: Frame rate
  :return: Number of frames
  """
  directory = save + "/images"
  if not os.path.exists(directory):
    os.makedirs(directory)

  def getFrame(sec, save_path):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
      x = "%04d.png" % (count,)
      thepath = save_path + "/theframe" + x
      print(thepath)
      cv2.imwrite(thepath, image)
    return hasFrames

  vidcap = cv2.VideoCapture(vid)
  sec = 0
  frameRate = fr  # //it will capture image in each 0.25 second
  count = 1
  success = getFrame(sec, directory)
  while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec, directory)
  return count


def main():
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument('--dir', help="Need a start video path", required=True)
  parser.add_argument('--save_path')
  parser.add_argument('--folder', type=int, help='Is this a folder of videos?', required=True)
  parser.add_argument('--frame_rate', type=float, help='Frame rate', default=0.25)


  args = parser.parse_args()

  # convert to dictionary
  frame_rate = args.frame_rate
  save_path = args.save_path

  if save_path and not os.path.exists(save_path):
    os.mkdir(save_path)

  # Single video.
  if args.folder == 0:
    if not save_path:
      save_path = args.dir[:-4] + "_frames/"
    split_video(args.dir, save_path, frame_rate)

  # A folder of mp4 videos.
  else:
    vids = listdir(args.dir)
    prefix = args.dir + "/"
    total_count = 0
    for f in vids:
      if not os.path.isdir(f) and f[-3:].lower() == "mp4":
        if not save_path:
          save_path = prefix + f[:-4] + "_frames/"
        else:
          save_path = save_path + "/" + f[:-4] + "_frames/"
        total_count += split_video(prefix + f, save_path, frame_rate)
    print(total_count)

if __name__ == "__main__":
  main()
