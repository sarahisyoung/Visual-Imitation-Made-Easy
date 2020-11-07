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
import io
from utils import jsonreader
import json
import torch.nn.functional as F
from utils import plot
import csv
from scipy.spatial import distance

import torchvision.models as models
import torch.nn as nn


def main():
  ######################################################################################
  ### Predict and visualize one model and one or more folders of images at a time. #####
  ######################################################################################

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_file', type=str, required=True)
  parser.add_argument('--grip_model_file', type=str, required=True)
  parser.add_argument('--image_folder', type=str, required=True)
  parser.add_argument('--output_folder', type=str, required=True)
  parser.add_argument('--folder', type=int, default=0)
  parser.add_argument('--which_gpu', default=0)


  args = parser.parse_args()

  # convert to dictionary
  params = vars(args)
  model_params = {}
  model_params['exp_name'] = "prediction"
  model_params['history'] = 1


  print("\n\n\nPredicting on " + params['model_file'] + "\n\n\n")
  print("\n\n\nSaving visualization to " + params['output_folder'] + "\n\n\n")


# =============================== STACKING MODEL ==========================================================================================
  s_params_json = "/".join(params['model_file'].split("/")[:-1]) + "/params.json"
  print("Using these stack params", s_params_json)
  with open(s_params_json) as json_file:
    s_model_params = json.load(json_file)
    device = torch.device('cuda:' + str(s_model_params['which_gpu']) if torch.cuda.is_available() else "cpu")
  s_model_params['task'] = "stack"
  s_model_params['lg'] = 1

  stack_model = TrashPolicy(s_model_params).to(device)

  s_dct = torch.load(params['model_file'], map_location=torch.device(device))
  if 'state_dict' in s_dct:
    s_dct = s_dct['state_dict']
  else:
    s_dct = s_dct
  try:
    stack_model.load_state_dict(s_dct)
  except Exception as e:
    print("\n========Some keys are missing, trying with strict=False========\n" + str(e))
    stack_model.load_state_dict(s_dct, strict=False)

  stack_model.eval()

  # =============================== GRIPPING MODEL ==========================================================================================

  # define models ================================================================
  model = models.resnet18(pretrained=True)

  # change last layer ============================================================
  model.fc = nn.Linear(in_features=512, out_features=1)

  # load the model ===============================================================
  print("loading the model")
  model.load_state_dict(torch.load(params['grip_model_file'], map_location=device))

  grip_model = model

  grip_model.eval()



  viz_folder = "/vizimages"
  savefolder = args.output_folder
  if not (os.path.exists(savefolder)):
    os.makedirs(savefolder)
  model_name = args.model_file[:-3].replace("/", "-")
  grip_model_name = args.grip_model_file[:-3].replace("/", "-")
  viz_folder = viz_folder +  "_" + grip_model_name + "/"
  viz_dir = savefolder + viz_folder + "/"
  if not (os.path.exists(viz_dir)):
    os.makedirs(viz_dir)

  print("\n******POSITIVE Y IS DOWN, POSITIVE X IS RIGHT, POSITIVE Z IS FORWARD*******\n")

  if args.folder == 0:
    imgs_folder = params['image_folder'].strip("/").split("/")[-1]  # local path i think
    total_t_error, total_a_error, _, total_imgs = predict_each_folder(params['image_folder'], imgs_folder, viz_dir, stack_model, grip_model, device)
    print("avg translation error:", total_t_error / total_imgs)
    print("avg angle error:", total_a_error / total_imgs)

  else:
    imgs_folder = params['image_folder']
    folders = sorted(os.listdir(imgs_folder))
    total_imgs = 0
    total_a_error = 0
    total_t_error = 0
    errors = []
    for f in folders:
      img_folder = params['image_folder'] + "/" + f + "/"
      the_imgs = sorted(glob.glob(img_folder + "/images/*"))
      if len(the_imgs) == 0:
        continue

      print("Predicting folder ", f)
      t_error, a_error, dir_err, num_imgs = predict_each_folder(img_folder, f, viz_dir, stack_model, grip_model, device)
      total_t_error += t_error
      total_a_error += a_error
      total_imgs += num_imgs
      print("avg translation error:", total_t_error / total_imgs)
      print("avg angle error:", total_a_error / total_imgs)





def get_img_from_fig(fig, dpi=180):
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=180)
  buf.seek(0)
  img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  buf.close()
  img = cv2.imdecode(img_arr, 1)
  return img

def get_pred_pics(imgname, save, trans, rot, realtrans, realrot, pred_grip):

  img_before = imgname

  plt = plot.plot_single(
    matplotlib.pyplot.imread(img_before),
    real_action_pos=realtrans, real_action_ang=realrot,
    gripper_logits=pred_grip,
    predicted_action_pos=trans, predicted_action_ang=rot,
    min_z=-5, max_z=5,
    img_name=imgname,
    discrete=True)
  img = get_img_from_fig(plt)
  img_number = imgname.split("/")[-1]
  status = cv2.imwrite(save + img_number, img)
  if status == False:
    print("didn't save img")
  plt.close()

def get_video(viz_dir_img, viz_dir, f):
  # FOR VIDEO
  writer, fps = None, 2
  # writer2 = None
  vizimgs = sorted(glob.glob(viz_dir_img + "/*"))
  for imgpath in vizimgs:
    img = cv2.imread(imgpath)
    if writer is None:
      w, h, ch = img.shape
      # writer = cv2.VideoWriter(viz_dir + "/" + f + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (h, w))
      writer = cv2.VideoWriter(viz_dir + "/" + f + ".mp4", 0x00000021, fps,  (h, w))
    writer.write(img)
    # writer2.write(img)

  cv2.destroyAllWindows()
  writer.release()
  # writer2.release()

def predict(stack_model, grip_model, img_t, device):
  rot = "6d-d"
  all_imgs = []
  all_imgs.append([img_t])

  input_size = 224
  transform = transforms.Compose([  # [1]
    transforms.Resize(250),  # [2]
    transforms.CenterCrop(input_size),  # [3]
    transforms.ToTensor(),  # [4]
    # transforms.Normalize(  # [5]
    #   mean=[0.485, 0.456, 0.406],  # [6]
    #   std=[0.229, 0.224, 0.225]  # [7]
    # )
  ])

  # define data loader ===========================================================
  mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

  grip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*mean_std)])

  for img_set in all_imgs:

    t_images = torch.unsqueeze(torch.unsqueeze(transform(Image.open(img_set[0])), 0),0) #only works for history=1
    t_grip_images = torch.unsqueeze(torch.unsqueeze(grip_transform(Image.open(img_set[0])), 0),0) #only works for history=1


    # save transformed images in same folder as original images
    for ii in img_set:
      splitted = ii.split("/")
      splitted[-1] = "t" +splitted[-1]
      splitted[-2] = "t_images"
      newfolder = "/".join(splitted[:-1])
      if not (os.path.exists(newfolder)):
        os.makedirs(newfolder)
      newname = "/".join(splitted)
      ttt = transform(Image.open(ii))
      torchvision.utils.save_image(ttt, newname)
    pred_trans, pred_ang, _ = stack_model.forward(t_images, rot)
    pred_rot = convert_6drep_to_mat(pred_ang, device)


    pred_mask = grip_model(t_grip_images.squeeze(0))
    prob_thr = [0.2, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99]
    p_thr = 0.5
    pred_prob = F.sigmoid(pred_mask)
    pred_mask_flat = pred_prob.view(-1)
    print("flat", pred_mask_flat)


    preds = pred_trans[0][0].tolist()
    print('PRED GRIPPER PROBABILITIES = ' + str(pred_mask_flat))
    pred_grip = pred_mask_flat.detach().numpy()
    rots = pred_rot[0]



    return preds, rots, pred_grip, newname

def get_real_labels(folder):
  imgs, real_rotations, real_translations = np.array([]), np.array([]), np.array([])
  if os.path.exists(folder + "/labels.json"):
    imgs, real_translations, real_rotations = jsonreader.get_vals(folder)
    scale_factor = np.max(np.abs(real_translations))
    real_translations = real_translations / scale_factor
    real_translations = real_translations / np.linalg.norm(real_translations, 1, 1)[:,None]
  else:
    print("No real images")
  return imgs, real_translations, real_rotations

def predict_each_folder(img_folder, saving_f, viz_dir, stack_model, grip_model, device):
  the_imgs = sorted(glob.glob(img_folder + "/images/*"))
  print("There are ", len(the_imgs), "images.")
  if len(the_imgs) == 0:
    print("there are no images. is the img folder correct?? ")
    return
  viz_dir_img = viz_dir + "__" + saving_f + "/"

  if not (os.path.exists(viz_dir_img)):
    os.makedirs(viz_dir_img)

  real_imgs, real_translations, real_rotations = get_real_labels(img_folder)

  if len(real_imgs) > 0:
    the_imgs = real_imgs
  total_trans_error = 0
  total_ang_error = 0
  total_dir_error = 0
  for i in range(len(the_imgs)):
    cur_img = the_imgs[i]
    # print("Predicting img ", cur_img)
    trans, rot, grip, new_path = predict(stack_model, grip_model, cur_img, device)
    if cur_img in real_imgs:
      real_trans, real_rots = real_translations[i], real_rotations[i]
      t_error = np.linalg.norm(real_trans - trans)
      total_trans_error += t_error
      a_error = np.linalg.norm(real_rots - rot.tolist())
      total_ang_error += a_error
      dir_err = distance.cosine(real_trans, trans)
      total_dir_error += dir_err

    else:
      real_trans, real_rots = [], []

    get_pred_pics(new_path, viz_dir_img, trans, rot, realtrans=real_trans, realrot=real_rots, pred_grip=grip)

  # FOR VIDEO
  get_video(viz_dir_img, viz_dir, saving_f)
  return total_trans_error, total_ang_error, total_dir_error, len(the_imgs)



if __name__ == "__main__":
  main()