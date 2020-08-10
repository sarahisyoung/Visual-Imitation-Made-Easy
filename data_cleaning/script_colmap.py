import os
import glob
import argparse
import subprocess
from split_video import split_video

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type=str)
  parser.add_argument('--gpu', type=str, default='4')
  parser.add_argument('--write', required=True)
  parser.add_argument('--use_mask', type=int, default=0)
  parser.add_argument('--split', type=int, help="Also split video into frames?", required=True)

  args = parser.parse_args()
  datapath = args.dir

  if not os.path.exists(args.write):
    os.mkdir(args.write)

  folders = sorted(glob.glob(datapath + "/*"), reverse=False)


  if args.split == 1:
    print("======== Splitting videos into frames ==========\n\n")
    endpath = args.dir + "/frames/"
    if not os.path.exists(endpath):
      os.mkdir(endpath)
    folders = sorted(glob.glob(args.dir + "/vids/*"), reverse=False)
    for video in folders:
      vid_name = video.strip("/").split("/")[-1]
      if not os.path.isdir(video) and vid_name[-3:].lower() == "mp4":
        split_video(video, endpath + vid_name[:-4] + "_frames/", 0.2)

    folders = sorted(glob.glob(args.dir + "/frames/*"), reverse=False)

  for f in folders:
      print("currently doing ", f)
      successes = []
      # COLMAP may fail due to lack of resources or other reasons.
      try:
        if args.use_mask:
          if not os.path.exists(f + "/masked_images"):
            successes.append(subprocess.call("python data_cleaning/mask.py --f 0 --dir " + f, shell=True))
          successes.append(subprocess.call("colmap feature_extractor --database_path " + f +
                                             "/database.db --image_path " + f + "/images --ImageReader.mask_path " +
                                             f + "/masked_images --SiftExtraction.gpu_index " +
                                             str(args.gpu), shell=True))

        else:
          successes.append(subprocess.call("colmap feature_extractor --database_path " + f +
                                   "/database.db --image_path " + f + "/images --SiftExtraction.gpu_index " + str(args.gpu), shell=True))

        successes.append(subprocess.call("colmap sequential_matcher --database_path " + f + "/database.db --SiftMatching.gpu_index " + str(args.gpu), shell=True))

        subprocess.call("mkdir " + f + "/sparse", shell=True)

        successes.append(subprocess.call("colmap mapper --database_path " + f + "/database.db --image_path "
                                         + f + "/images --output_path " + f + "/sparse", shell=True))
        subprocess.call("mkdir " + f + "/dense", shell=True)

        successes.append(subprocess.call("colmap image_undistorter \
                          --image_path " + f + "/images \
                              --input_path " + f + "/sparse/0 \
                                  --output_path " + f + "/dense \
                                      --output_type COLMAP \
                                          --max_image_size 2000", shell=True))

        successes.append(subprocess.call("colmap patch_match_stereo \
                          --workspace_path " + f + "/dense \
                              --workspace_format COLMAP \
                                  --PatchMatchStereo.geom_consistency true --PatchMatchStereo.gpu_index " + str(args.gpu), shell=True))
        successes.append(subprocess.call("colmap stereo_fusion \
                          --workspace_path " + f + "/dense \
                              --workspace_format COLMAP \ --input_type geometric \
                                  --output_path " + f + "/dense/fused.ply", shell=True))

        successes.append(subprocess.call(
          "colmap model_converter --input_path " + f + "/sparse/0/ --output_path " + f + "/sparse/0/ --output_type TXT", shell=True))
        print("---------------------Writing json file now.------------------------")
        successes.append(subprocess.call("python data_cleaning/write_translations.py --dir " + f, shell=True))
        print("---------------------Doing labels tester now.----------------------")
        successes.append(subprocess.call("python data_cleaning/labels_tester.py --dir " + f + " --write " + args.write, shell=True))
        print("----------------------Removing dense/stereo now.--------------------")
        successes.append(subprocess.call("rm -r " + f + "/dense/stereo", shell=True))


        if any(successes) != 0:
          print("RIP", successes)
          return

      except subprocess.CalledProcessError as e:
        print("process error: ", e)
        return
      except OSError as e:
        print("OS got an error oof: ", e)
        return

main()
