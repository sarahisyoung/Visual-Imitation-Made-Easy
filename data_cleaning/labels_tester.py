import glob
import os
import io
import numpy as np
import time
import matplotlib
import cv2
from utils import jsonreader, plot
from utils.utils import get_img_from_fig
from torchvision import transforms
from PIL import Image

plot_transform = transforms.Compose([  # [1]
    transforms.Resize(224),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
])


def get_updated_pics(f, save, file="/labels.json"):
    # for video
    writer, fps = None, 3
    file = f + file
    folder = f.strip("/").split("/")[-1]
    if not os.path.exists(file):
        print("FOLDER " + f + " doesn't have " + file + " file.")
        return

    the_imgs = sorted(glob.glob(f + "/images/*"))
    num_real_imgs = len(the_imgs)

    # Only returns images that actually exist.
    imgs, trans, rot = jsonreader.get_vals(f)

    scale_factor = np.max(np.abs(trans))
    if scale_factor > 4 or scale_factor < 0.1:
        print((np.where(trans == 1)))
        print("Max norm(action) is not normal: ", scale_factor, "for", imgs[(np.where(trans ==1))[0]])

    if num_real_imgs != len(imgs)+1:
        print("WARNING: number of colmap imges != total number of images. Colmap: ", len(imgs), "total: ", num_real_imgs)

    for i in range(len(imgs)):
        img_before_name = imgs[i]
        img_before = plot_transform(Image.open(img_before_name)).permute(1, 2, 0).numpy()
        plt = plot.plot_single(
            img_before,
            real_action_pos=trans[i],
            real_action_ang=rot[i],
            img_name=img_before_name,
            min_z=-5, max_z=5,
            verbose=True,
            )

        img = get_img_from_fig(plt)
        plt.close()

        if writer is None:
            w, h, ch = img.shape
            video_name = save + "/" + time.strftime("%d-%m-%Y_%H-%M-%S") + "_" + folder + "thelabeltest.avi"
            writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (h, w))

        writer.write(img)

    cv2.destroyAllWindows()
    writer.release()
    print("Done with " + f)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', required=True)
    parser.add_argument('--write')
    parser.add_argument('--folder', type=int, default=0)

    args = parser.parse_args()
    dir = args.dir

    # convert to dictionary
    params = vars(args)
    towrite = params['write']

    if params['folder'] == 0:

        if not towrite:
            towrite = dir.strip("/").split('/')[-1] + "_video"
            print("writing to", dir + "/" + towrite)
            if not os.path.exists(dir + "/" + towrite):
                os.mkdir(dir + "/" + towrite)
            get_updated_pics(dir, dir + "/" + towrite)
        else:
            get_updated_pics(dir, towrite)

    else:
        folders = sorted(os.listdir(dir))
        if not towrite:
            if not os.path.exists(dir + "/check_labels/"):
                os.mkdir(dir + "/check_labels/")
            for f in folders:
                print("FOLDER IS: " + dir + f + " and will be saved as " + f + "_video")
                if not os.path.exists(dir + "/check_labels/" +f+ "_video"):
                    os.mkdir(dir + "/check_labels/"+f+ "_video")
                get_updated_pics(dir + "/" + f, dir + "/check_labels/"+f+ "_video")
        else:
            for f in folders:
                print("FOLDER IS: " + dir + f + " and will be saved to " + towrite)
                get_updated_pics(dir + "/" + f, towrite)



if __name__ == "__main__":
  main()
