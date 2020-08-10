import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from base import BaseModel



class TrashPolicy(BaseModel):
  out_features = 400
  mse_loss = nn.MSELoss()
  l1_loss = nn.L1Loss()
  train_test = 0.8
  dimm = 512
  sigmoid = torch.nn.Sigmoid()
  cross_entropy_loss = nn.CrossEntropyLoss()

  def __init__(self, params):
    super(TrashPolicy, self).__init__()
    torch.manual_seed(0)
    np.random.seed(0)

    self.device = torch.device('cuda:' + str(params['which_gpu']) if torch.cuda.is_available() else "cpu")
    self.feat_dim = params['feat_dim']
    self.pretrained = params['pretrained']
    self.history = params['history']
    self.l1 = params['l1']
    self.l2 = params['l2']
    self.l3 = params['l3']
    self.lg = params['lg']
    self.rad = params['rad']
    self.task = params['task']
    self.scaling = {}
    self.z_dim = 256
    self.net_type = params['net_type']


    # Part 1: first 5 layers of alexnet + extra layer C200
    if self.net_type == "alex":
      self.dimm = 256
      alexnet = models.alexnet(pretrained=self.pretrained).to(self.device)
      alxlt = list(alexnet.features)
      new_conv_layer = nn.Conv2d(256, self.dimm, kernel_size=(3, 3), stride=2, padding=1).to(self.device)
      new_max_pool_layer = nn.MaxPool2d(kernel_size=(7, 7)).to(self.device)
      new_features = nn.Sequential(
        alxlt[0],
        alxlt[1],
        alxlt[2],
        alxlt[3],
        alxlt[4],
        alxlt[5],
        alxlt[6],
        alxlt[7],
        alxlt[8],
        alxlt[9],
        alxlt[10],
        alxlt[11],
        new_conv_layer, nn.ReLU(), new_max_pool_layer).to(self.device)

    self.image_feat = new_features.to(self.device)


    # POSITION NET
    pfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim, bias=True).to(self.device) # *3 for 3 images
    pfc2 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)
    self.p_net = nn.Sequential(pfc1, nn.ReLU(), pfc2).to(self.device)

    if self.net_type=="alex":
      pfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim * 2, bias=True).to(self.device)  # *3 for 3 images
      pfc2 = nn.Linear(in_features=self.feat_dim * 2, out_features=self.feat_dim, bias=True).to(self.device)
      pfc3 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)  # *3 for 3 images

      self.p_net = nn.Sequential(pfc1, nn.ReLU(), pfc2, nn.ReLU(), pfc3).to(self.device)




    # ANGLE NETS

    # INDEPENDENT, 6D: input 3 images + pos (3) concatenated, output angle in 6d representation
    in_a6dfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
    in_a6dfc2 = nn.Linear(in_features=self.feat_dim, out_features=6, bias=True).to(self.device)  # 2 * 3 matrix
    self.in_a6d_net = nn.Sequential(in_a6dfc1, nn.ReLU(), in_a6dfc2).to(self.device).to(self.device)

    # INDEPENDENT, 3D: input 3 images + pos (3) concatenated, output angle in angle-axis representation
    in_a3dfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
    in_a3dfc2 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)  # 1 * 3 matrix
    self.in_a3d_net = nn.Sequential(in_a3dfc1, nn.ReLU(), in_a3dfc2).to(self.device).to(self.device)

    # DEPENDENT, 6D: input 3 images + pos (3) concatenated, output angle in 6d representation
    dep_a6dfc1 = nn.Linear(in_features=3 + self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
    dep_a6dfc2 = nn.Linear(in_features=self.feat_dim, out_features=6, bias=True).to(self.device) # 2 * 3 matrix
    self.dep_a6d_net = nn.Sequential(dep_a6dfc1, nn.ReLU(), dep_a6dfc2).to(self.device).to(self.device)

    # DEPENDENT, 3D: input 3 images + pos (3) concatenated, output angle in angle-axis representation
    dep_a3dfc1 = nn.Linear(in_features=3 + self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
    dep_a3dfc2 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)  # 1 * 3 matrix
    self.dep_a3d_net = nn.Sequential(dep_a3dfc1, nn.ReLU(), dep_a3dfc2).to(self.device).to(self.device)


  def forward(self, img_lst, rot="6d-d", curl=False, detach=False):

    cl_out_pos_lst, argmax_p_lst, cl_out_angle_lst, argmax_a_lst, cl_out_len_lst, gripper_status = [], [], [], [], [], []

    if curl==True:

      # img_lst shape: [batch, 1, 3, 224, 224]
      for imgs in img_lst:  # for each item in batch
        t_i = imgs
        t_i = t_i.to(self.device)
        img_ti_feat = self.image_feat(t_i).view(1, -1).to(self.device)

        cl_out_pos_lst.append(img_ti_feat)
      z_out = torch.cat(cl_out_pos_lst)
      if detach:
        z_out = z_out.detach()

      return z_out

    for imgs in img_lst:
      feats = []
      t_i = imgs
      t_i = t_i.to(self.device)
      img_ti_feat = self.image_feat(t_i).view(1, -1).to(self.device)
      feats.append(img_ti_feat)

      combined_imgs = torch.cat((feats), dim=1).to(self.device)
      cl_out_pos = self.p_net(combined_imgs).to(self.device)
      cl_out_pos_lst.append(cl_out_pos)


      if rot == "6d-d": # dependent
        combined_imgs_pos = torch.cat((combined_imgs, cl_out_pos), dim=1).to(self.device)
        cl_out_angle = self.dep_a6d_net(combined_imgs_pos).to(self.device)
        cl_out_angle_lst.append(cl_out_angle)
      elif rot == "axis-d": # dependent
        combined_imgs_pos = torch.cat((combined_imgs, cl_out_pos), dim=1).to(self.device)
        cl_out_angle = self.dep_a3d_net(combined_imgs_pos).to(self.device)
        cl_out_angle_lst.append(cl_out_angle)
      elif rot == "6d-i":
        cl_out_angle = self.in_a6d_net(combined_imgs).to(self.device)
        cl_out_angle_lst.append(cl_out_angle)
      elif rot == "axis-i": # independent
        cl_out_angle = self.in_a3d_net(combined_imgs).to(self.device)
        cl_out_angle_lst.append(cl_out_angle)
      else:
        raise Exception("Rotation must be: 6d-d, 6d-i, axis-d, or axis-i")

    if len(gripper_status) == 0:
      return cl_out_pos_lst, cl_out_angle_lst, []
    return cl_out_pos_lst, cl_out_angle_lst, torch.cat(gripper_status).to(self.device)






