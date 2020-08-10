from torchvision import transforms
from models.forward_model import TrashPolicy

import torch
from utils.utils import convert_6drep_to_mat
from PIL import Image
import torchvision
import glob
import matplotlib
import os
import numpy as np
import cv2
from utils.utils import get_img_from_fig
from utils import jsonreader
import json
from utils import plot
from scipy.spatial import distance



def main():
  ######################################################################################
  ### Predict and visualize one model and one or more folders of images at a time. #####
  ######################################################################################

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_file', type=str, required=True)
  parser.add_argument('--image_folder', type=str, required=True)
  parser.add_argument('--output_folder', type=str, required=True)
  parser.add_argument('--folder', type=int, default=0)
  parser.add_argument('--which_gpu', default=0)


  args = parser.parse_args()

  # convert to dictionary
  params = vars(args)


  print("\n\n\nPredicting on " + params['model_file'] + "\n\n\n")
  print("\n\n\nSaving visualization to " + params['output_folder'] + "\n\n\n")

  params_json = "/".join(params['model_file'].split("/")[:-1]) + "/params.json"
  print("Using these params", params_json)
  with open(params_json) as json_file:
    model_params = json.load(json_file)
    device = torch.device('cuda:' + str(model_params['which_gpu']) if torch.cuda.is_available() else "cpu")

  ##################################
  ### CREATE DIRECTORY FOR LOGGING
  ##################################
  model_params['task'] = "push"
  model = TrashPolicy(model_params).to(device)
  d = torch.load(params['model_file'], map_location=torch.device(device))
  if 'state_dict' in d:
    dct = d['state_dict']
  else:
    dct = d
  try:
    model.load_state_dict(dct)
  except Exception as e:
    print("\n========Some keys are missing, trying with strict=False========\n" + str(e))
    model.load_state_dict(dct, strict=False)

  viz_folder = "/vizimages"
  savefolder = args.output_folder
  if not (os.path.exists(savefolder)):
    os.makedirs(savefolder)
  model_name = args.model_file[:-3].replace("/", "-")
  viz_folder = viz_folder + model_name + "/"
  viz_dir = savefolder + viz_folder + "/"
  if not (os.path.exists(viz_dir)):
    os.makedirs(viz_dir)

  print("\n******POSITIVE Y IS DOWN, POSITIVE X IS RIGHT, POSITIVE Z IS FORWARD*******\n")

  if args.folder == 0:
    imgs_folder = params['image_folder'].strip("/").split("/")[-1]  # local path i think
    predict_each_folder(params['image_folder'], imgs_folder, viz_dir, model, device)


  else:
    imgs_folder = params['image_folder']
    folders = sorted(os.listdir(imgs_folder))
    for f in folders:
      img_folder = params['image_folder'] + "/" + f + "/"
      the_imgs = sorted(glob.glob(img_folder + "/images/*"))
      if len(the_imgs) == 0:
        continue

      print("Predicting folder ", f)
      predict_each_folder(img_folder, f, viz_dir, model, device)

def get_pred_pics(imgname, save, trans, rot):
  img_before = imgname
  plt = plot.plot_single(
    matplotlib.pyplot.imread(img_before),
    real_action_pos=[], real_action_ang=[],
    predicted_action_pos=trans, predicted_action_ang=rot,
    img_name=imgname)

  img = get_img_from_fig(plt)
  img_number = imgname.split("/")[-1]
  status = cv2.imwrite(save + img_number, img)
  if status == False:
    print("didn't save img")
  plt.close()

def get_video(viz_dir_img, viz_dir, f):
  # FOR VIDEO
  writer, fps = None, 2
  vizimgs = sorted(glob.glob(viz_dir_img + "/*"))
  for imgpath in vizimgs:
    img = cv2.imread(imgpath)
    if writer is None:
      w, h, ch = img.shape
      writer = cv2.VideoWriter(viz_dir + "/" + f + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (h, w))
    writer.write(img)
  cv2.destroyAllWindows()
  writer.release()

def predict(model, img_t, device):
  rot = "6d-d"

  input_size = 224
  transform = transforms.Compose([  # [1]
    transforms.Resize(250),  # [2]
    transforms.CenterCrop(input_size),  # [3]
    transforms.ToTensor(),  # [4]
  ])


  t_images = torch.unsqueeze(torch.unsqueeze(transform(Image.open(img_t)), 0),0) #only works for history=1

  # save transformed images in same folder as original images
  splitted = img_t.split("/")
  splitted[-1] = "t" +splitted[-1]
  splitted[-2] = "t_images"
  newfolder = "/".join(splitted[:-1])
  if not (os.path.exists(newfolder)):
    os.makedirs(newfolder)
  newname = "/".join(splitted)
  ttt = transform(Image.open(img_t))
  torchvision.utils.save_image(ttt, newname)

  pred_trans, pred_ang, _ = model.forward(t_images, rot)
  pred_rot = convert_6drep_to_mat(pred_ang, device)


  preds = pred_trans[0][0].tolist()
  rots = pred_rot[0]
  return preds, rots, newname


def predict_each_folder(img_folder, saving_f, viz_dir, model, device):
  the_imgs = sorted(glob.glob(img_folder + "/images/*"))
  print("There are ", len(the_imgs), "images.")
  viz_dir_img = viz_dir + "__" + saving_f + "/"

  if not (os.path.exists(viz_dir_img)):
    os.makedirs(viz_dir_img)


  for i in range(len(the_imgs)):
    cur_img = the_imgs[i]
    trans, rot, new_path = predict(model, cur_img, device)
    get_pred_pics(new_path, viz_dir_img, trans, rot)

  # FOR VIDEO
  get_video(viz_dir_img, viz_dir, saving_f)
  return



if __name__ == "__main__":
  main()
