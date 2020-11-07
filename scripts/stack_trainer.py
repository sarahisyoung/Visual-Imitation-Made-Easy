import os
import utils.plot as plotting
import utils.logger as logger
import torch
import numpy as np
import dataloader.StackDataset as StackDataset
import torch.nn as nn

from utils.utils import convert_6drep_to_mat, convert_3dmat_to_6d_rep, get_img_from_fig
from torch import optim
from utils.earlystop import EarlyStop
import wandb
import io
import cv2

import torchvision.models

class StackTrainer:
  """
  Trainer class
  """

  def __init__(self, model, params):

    self.params = params
    self.device = torch.device('cuda:' + str(params['which_gpu']) if torch.cuda.is_available() else "cpu")
    self.model = model
    self.logger = logger.Logger(self.params['logdir'])
    self.batch_size = params['batch_size']
    self.epochs = params['epochs']
    self.rot = params['rot']
    self.model_name = params['model']
    self.history = params['history']
    self.mult = params['mult']
    self.l1 = params['l1']
    self.l2 = params['l2']
    self.l3 = params['l3']
    self.l4 = params['l4']
    self.l5 = params['l5']
    self.l6 = params['l6']
    self.l7 = params['l7']
    self.l8 = params['l8']
    self.rad = params['rad']
    self.net = params['net_type']
    self.task = params['task']
    self.lg = params['lg']

    self.baseline = 0
    if 'baseline' in params:
      self.baseline = params['baseline']
    self.mse_loss = nn.MSELoss()
    self.scaled_mse_loss = nn.MSELoss(reduce=False)
    self.l1_loss = nn.L1Loss()


    self.train_set = StackDataset.StackDataset("train", params)
    self.num_train = len(self.train_set)
    self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                                    sampler=None,
                                                    batch_sampler=None, num_workers=6,
                                                    # collate_fn=self.train_set.my_collate,
                                                    pin_memory=False, drop_last=False, timeout=0,
                                                    worker_init_fn=None)
    self.val_set = StackDataset.StackDataset("val", params)
    self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True,
                                                   # collate_fn=self.val_set.my_collate,
                                                   num_workers=6,
                                                   pin_memory=False, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
    self.num_val = len(self.val_set)

    self.test_set = StackDataset.StackDataset("test", params)
    self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True,
                                                  # collate_fn=self.val_set.my_collate,
                                                  num_workers=6,
                                                  pin_memory=False, drop_last=False, timeout=0,
                                                  worker_init_fn=None)
    self.num_test = len(self.test_set)


    wandb.init(project="trashbot")
    wandb.save("*.pt")
    params['num_train_runs'] = self.num_train
    params['num_val_runs'] = self.num_val
    params['num_test_runs'] = self.num_test
    wandb.config.update(params)

    self.optimizer = optim.Adam(model.parameters(), lr=self.params['lr'])
    frozen = ["image_feat", "W"]
    if params['ckp']:
      ckp_path = params['ckp']
      checkpoint = torch.load(ckp_path, map_location=self.device)
      if 'zdim' in params:
        self.model.W = nn.Parameter(torch.rand(params['zdim'], params['zdim']))
      model_dict = self.model.state_dict()
      pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in frozen}

      self.model.load_state_dict(pretrained_dict, strict=False)

    if params['finetune'] == 1 and params['rad'] != 'simclr':

      for name, param in self.model.named_parameters():
        if frozen[0] in name or name == frozen[1]:
          print("Freezing this", name)
          param.requires_grad = False

      print("\n\nTo be trained: \n\n")
      for name, param in self.model.named_parameters():
        if param.requires_grad:
          print(name)

      self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.params['lr'])

    model = torchvision.models.resnet18(pretrained=True)

    # change last layer ============================================================
    model.fc = nn.Linear(in_features=512, out_features=1)

    # load the model ===============================================================
    print("loading the grip model")
    model.load_state_dict(torch.load(params['grip_file'], map_location=self.device))
    self.grip_model = model.to(self.device)
    self.grip_model.eval()

  def train(self, logdir=None, grad_clip=None):
    wandb.watch(self.model)
    # self.model.train()
    # if self.rad == "curl":
    #   self.CURL.train(True)
    # weight_decay = 0.01
    self.num_train_batches = np.ceil(self.num_train / self.batch_size)

    early_stopping = EarlyStop(patience=150)
    num_epochs = self.epochs
    for epoch in range(num_epochs):  ## run the model for 10 epochs
      self.wandb_dict = {}
      ######### TRAIN ===============================================================================================
      self.model.train()
      self.run_loop(self.train_loader, "train", epoch, logdir, None)


      ######### VALIDATE ============================================================================================
      self.model.eval()
      with torch.no_grad():
        early_stop = self.run_loop(self.val_loader, "val", epoch, logdir, early_stopping)


      ######### TEST ============================================================================================
      self.model.eval()
      with torch.no_grad():
        self.run_loop(self.test_loader, "test", epoch, logdir, None)

      wandb.log(self.wandb_dict)


      if early_stop:
        print("Early stopping at epoch " + str(epoch))


  def run_loop(self, data_loader, current, epoch, logdir, early_stopping):

    epoch_error = []
    epoch_loss = []

    pos_loss = []
    pos_error = []

    ang_loss = []
    ang_error = []

    gripper_loss = []
    gripper_error = []

    dir_loss = []
    scaledl2_loss = []

    self.model.scaling = {}

    for i_batch, (batch_img_names, batch_imgs, batch_actions, gripper_labels, runs, aug) in enumerate(data_loader):
      batch_imgs = batch_imgs.to(self.device)
      batch_actions.to(self.device)

      gripper_labels = gripper_labels.to(self.device).float()

      real_positions, real_angles, real_angles_raw = self.get_label_act_pos(batch_actions)

      real_angles_raw = real_angles_raw.to(self.device)
      # real_angles = real_angles.to(self.device)
      real_positions = real_positions.to(self.device)

      # pred_pos, pred_ang, pred_gripper = self.model.forward(batch_imgs, self.rot)

      if self.baseline == 1:
        p_lst = []
        a_lst = []
        for i in range(len(batch_img_names)):
          rand_tens = torch.rand((1, 3))
          ang = torch.rand((1, 6))
          a_lst.append(ang)
          p_lst.append(rand_tens / np.linalg.norm(rand_tens, 1))
        pred_pos = p_lst
        pred_ang = a_lst
        pred_gripper = [0] * len(batch_img_names)
      else:
        pred_pos, pred_ang, pred_gripper = self.model.forward(batch_imgs, self.rot)
        pred_gripper = self.grip_model(batch_imgs.squeeze(1))

      total_loss, total_error, p_loss, p_error, a_loss, a_error, g_loss, g_error, d_loss, sl2_loss = self.calculate_loss_error(pred_pos, pred_ang,
                                                                                             real_positions, real_angles,
                                                                                             self.mult, runs,
                                                                                             pred_gripper, gripper_labels)

      # if p_error > 0.2 and epoch > 2 and current == "train":
      #   print("YO WHAT BIG Error", batch_img_names, p_loss)
      if current == "train" and self.baseline != 1:
        self.optimizer.zero_grad()
        # 3. backward propagation
        total_loss.backward()

        # 4. weight optimization
        self.optimizer.step()


      epoch_loss.append(total_loss.item())
      epoch_error.append(total_error.item())

      ang_loss.append(a_loss.item())
      ang_error.append(a_error.item())

      pos_loss.append(p_loss.item())
      pos_error.append(p_error.item())

      dir_loss.append(d_loss.item())
      scaledl2_loss.append(sl2_loss.item())

      if self.lg != 0:
        gripper_loss.append(g_loss.item())
        gripper_error.append(g_error.item())
      else:
        gripper_loss.append(0)
        gripper_error.append(0)



      pred_3d_ang = pred_ang
      if "6d" in self.rot and self.baseline != 1:
        pred_3d_ang = convert_6drep_to_mat(pred_ang, self.device)

      if i_batch % 100 == 0 and current == "train":
        print('Epoch [{}/{}, Step [{}/{}], Loss: {:.4f}'
              .format(epoch, self.epochs, i_batch + 1, self.num_train_batches, total_loss.item()))

      self.log_sometimes(epoch, i_batch, batch_img_names, batch_imgs, real_positions, real_angles_raw, gripper_labels,
                         pred_gripper, pred_pos, pred_3d_ang, logdir, current)


    print("Epoch", epoch, current + " Error", np.mean(epoch_error))
    print("Epoch", epoch, current + " Loss", np.mean(epoch_loss))
    print("Epoch", epoch, current + " Gripper Loss: ", np.mean(gripper_loss))

    self.wandb_dict.update({
      current.lower().capitalize() + " Error": np.mean(epoch_error),
      current.lower().capitalize() + " Loss": np.mean(epoch_loss),

      current.lower().capitalize() + " Translation Loss": np.mean(pos_loss),
      current.lower().capitalize() + " Translation Error": np.mean(pos_error),

      current.lower().capitalize() + " Angle Loss": np.mean(ang_loss),
      current.lower().capitalize() + " Angle Error": np.mean(ang_error),

      current.lower().capitalize() + " Gripper Loss": np.mean(gripper_loss),
      current.lower().capitalize() + " Gripper Error": np.mean(gripper_error),

      current.lower().capitalize() + " Direction Loss": np.mean(dir_loss),

      current.lower().capitalize() + " Scaledl2 Loss": np.mean(scaledl2_loss),

    })


    self.logger.log_scalar(np.mean(epoch_error), current + " Error", epoch)
    self.logger.log_scalar(np.mean(epoch_loss), current + " Loss", epoch)

    self.logger.log_scalar(np.mean(pos_loss), current + " Translation Loss", epoch)
    self.logger.log_scalar(np.mean(pos_error), current + " Translation Error", epoch)

    self.logger.log_scalar(np.mean(ang_loss), current + " Angle Loss", epoch)
    self.logger.log_scalar(np.mean(ang_error), current + " Angle Error", epoch)

    self.logger.log_scalar(np.mean(gripper_loss), current + " Gripper Loss", epoch)
    self.logger.log_scalar(np.mean(gripper_error), current + " Gripper Error", epoch)

    if current == "train":
      if epoch % 25 == 0 or epoch >= self.epochs - 1:
        self.save_model_all(logdir, self.model_name, epoch)

    elif current == "val":
      stop, save = early_stopping(np.mean(epoch_loss), self.model, epoch)
      if save:
        self.save_model_all(logdir, self.model_name, epoch, earlystop=True)
      return stop

  def save_model_all(self, save_dir, model_name, epoch, earlystop=False):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param model_name:  model name
    :param epoch:  epoch
    :return:  None
    """
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    if earlystop:
      save_path = '{}_earlystop.pt'.format(save_prefix)
    print("Saving model to {}".format(save_path))
    output = open(save_path, mode="wb")

    checkpoint = {
      'epoch': epoch,
      'state_dict': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }

    torch.save(checkpoint, output)
    torch.save(checkpoint, os.path.join(wandb.run.dir, 'model.pt'))

    # torch.save(model.state_dict(), save_path)
    output.close()

  def plot(self, batch_img_names, batch_imgs, batch_pos, batch_ang, real_gripper, pred_gripper, predicted_pos, predicted_ang, epoch, batch, logdir, current, log=True):
    error = []
    for j in range(len(batch_img_names)):
      img1_name = batch_img_names[j]
      # don't save ALL the pictures, takes too much space. only save *0_frames, *4_frames, *7_frames,  *9_frames and unmirrored
      # if "mirror" in img1_name or not ("0_frames" in img1_name or "4_frames" in img1_name or "7_frames" in img1_name or "9_frames" in img1_name):
      #   continue
      real_action_pos = batch_pos[j]
      real_action_ang = batch_ang[j]
      predicted_action_pos = predicted_pos[j][0]
      predicted_action_ang = predicted_ang[j]


      if self.history == 1:
        img1 = batch_imgs[j]

        if self.task == "stack":                                # plot with gripper
          g = real_gripper[j]
          gripper_logits = pred_gripper[j]
          plott = plotting.plot_single(
            img1.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
            real_action_pos.detach().cpu(), real_action_ang.detach().cpu(),
            g.detach().cpu(),
            gripper_logits.detach().cpu(),
            predicted_action_pos.detach().cpu(), predicted_action_ang,
            img_name=img1_name, min_z=-1, max_z=1)
        else:                                                   # plot without gripper
          plott = plotting.plot_single(
            img1.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
            real_action_pos.detach().cpu(), real_action_ang.detach().cpu(),
            predicted_action_pos=predicted_action_pos.detach().cpu(),
            predicted_action_ang=predicted_action_ang,
            img_name=img1_name, min_z=-1, max_z=1)

      folders = img1_name.split("/")
      img_loc = folders[-3] + "_" + folders[-1].split(".")[0]

      if "mirror" not in img1_name and (
              "30_frames" in img1_name or "89_frames" in img1_name or "44_frames" in img1_name):
        k = current.lower().capitalize() + " images"
        # plott.show()
        # if k not in self.wandb_dict:
        #   self.wandb_dict[k] = [wandb.Image(get_img_from_fig(plott), caption=img1_name)]
        # else:
        #   self.wandb_dict[k].append(wandb.Image(get_img_from_fig(plott), caption=img1_name))


      if current == "val":
        plott.savefig(logdir + '/valimages/' + img_loc)
        if log:
          self.logger.log_figure(plott.figure(1), "Predicted vs Actual Val", batch * len(batch_img_names) + j, epoch)
      elif current == "test":
        plott.savefig(logdir + '/testimages/' + img_loc)
      else:
        plott.savefig(logdir + '/trainimages/' + img_loc)
        if log:
          self.logger.log_figure(plott.figure(1), "Predicted vs Actual Train", batch * len(batch_img_names) + j, epoch)

      plott.close()


  def log_sometimes(self, epoch, i_batch, batch_img_names, batch_imgs, real_positions, real_angles_raw, real_gripper, pred_gripper, pred_pos, pred_ang, logdir, val):
    if epoch == -1:
      self.plot(batch_img_names, batch_imgs, real_positions, real_angles_raw, real_gripper, pred_gripper, pred_pos, pred_ang, epoch, i_batch, logdir, current=val,
                log=False)
    if epoch % 10 == 0:
      if i_batch % 100 == 0 and epoch % 20:
        self.plot(batch_img_names, batch_imgs, real_positions, real_angles_raw, real_gripper, pred_gripper, pred_pos, pred_ang, epoch, i_batch, logdir, current=val, log=True)
      else:
        self.plot(batch_img_names, batch_imgs, real_positions, real_angles_raw, real_gripper, pred_gripper, pred_pos, pred_ang, epoch, i_batch, logdir, current=val, log=False)
    elif epoch == self.epochs - 1:
      self.plot(batch_img_names, batch_imgs, real_positions, real_angles_raw, real_gripper, pred_gripper, pred_pos, pred_ang, epoch, i_batch, logdir, current=val,
                log=True)


  def get_label_act_pos(self, batch_actions):
    """
    :param batch_actions:
    :return:  real positions: tensor of size (batch, 3)
              real_angles: list of tensors of size 6 (batch, 6)
              raw_angles: tensor of size (batch, 3, 3)
    """

    # 1st row of every batch is positions (x,y,z), rest is angle
    real_positions = batch_actions[:, 0]
    real_angles_raw = batch_actions[:, 1:]
    real_angles = real_angles_raw

    if "6d" in self.rot:
      real_angles = convert_3dmat_to_6d_rep(real_angles_raw, self.device)
    return real_positions, real_angles, real_angles_raw

  def scale_invariant_l2(self, pred, real, runs):
    """
    :param pred: tensor of shape (batch, 3)
    :param real: tensor of shape (batch, 3)
    :return: loss
    """
    scaling = self.model.scaling
    scaled_pred = torch.clone(pred).to(self.device)
    batch = len(pred)
    for i in range(batch):
      curr_traj = runs[i]
      curr_top = pred[i].dot(real[i]).item()
      curr_denom = torch.dot(pred[i], pred[i]).item()
      if curr_traj in scaling:
        scaling[curr_traj][0] += curr_top
        scaling[curr_traj][1] += curr_denom
      else:
        scaling[curr_traj] = [curr_top, curr_denom]
      s = scaling[curr_traj][0] / scaling[curr_traj][1]
      scaled_pred[i] = s * pred[i]

    return self.mse_loss(scaled_pred, real)

  def direction_loss(self, pred, real):
    # only look at 0th and last column (x and z since y doesn't matter as much)
    real = torch.stack((real[:, 0], real[:, -1]), 1).to(self.device)
    pred = torch.stack((pred[:, 0], pred[:, -1]), 1).to(self.device)
    top = torch.bmm(real.view(real.shape[0], 1, real.shape[-1]), pred.view(pred.shape[0], pred.shape[-1], 1)).to(
      self.device)
    top = top.squeeze(1).squeeze(1)
    bottom = torch.norm(real, dim=1) * torch.norm(pred, dim=1)
    eps = 1e-7
    arcd = torch.clamp(top / bottom, min=-1 + eps, max=1 - eps)
    return torch.mean(torch.acos(arcd)).to(self.device)

  def calculate_loss_error(self, pred_pos, pred_ang, real_positions, real_angles, mult, runs, pred_grip=None, real_grip=None):
    p_tensor = torch.stack(pred_pos).reshape(len(pred_pos), -1).to(self.device)
    a_tensor = torch.stack(pred_ang).reshape(len(pred_ang), -1).to(self.device)

    label_pos = np.array([l.tolist() for l in real_positions]).reshape(len(real_positions), -1)
    label_ang = np.array([l.tolist() for l in real_angles]).reshape(len(real_angles), -1)

    label_pos = torch.from_numpy(label_pos).float().to(self.device)
    label_ang = torch.from_numpy(label_ang).float().to(self.device)


    mse_ploss = self.mse_loss(p_tensor, label_pos).to(self.device)
    mse_aloss = self.mse_loss(a_tensor, label_ang).to(self.device)

    l1_ploss = self.l1_loss(p_tensor, label_pos)

    scaled_l2_loss = self.scale_invariant_l2(p_tensor, label_pos, runs).to(self.device)



    d_loss = self.direction_loss(p_tensor, label_pos).to(self.device)


    # ================================================ LOSSES =================================================
    #       MSE loss: coef l2
    #       Balanced MSE loss: coef l4, MSE_loss divided by P(x in bucket), where there are 3 equal buckets in (-1,1)
    #       L1 Loss: coef l1
    #       direction loss: coef l3


    sign_loss = 0

    ### NORM LOSS ####
    norms = torch.norm(p_tensor, dim=1)

    norm_loss = torch.mean((norms-1)**2)



    p_loss = self.l1 * l1_ploss + \
            self.l2 * mse_ploss + \
            self.l3 * d_loss + \
            self.l5 * sign_loss + \
            self.l6 * norm_loss + \
            self.l7 * scaled_l2_loss \


    aloss = mse_aloss
    a_error = mse_aloss

    p_error = mse_ploss

    g_loss = 0
    g_error = 0

    if self.task == "stack" and self.lg != 0:
      gripper_labels = real_grip.unsqueeze(1)
      g_loss = self.l1_loss(pred_grip, gripper_labels)
      g_error = g_loss

    total_loss = p_loss + (aloss * mult) + (self.lg * g_loss)
    total_error = p_error + (a_error * mult) + (self.lg * g_error)

    return total_loss, total_error, p_loss, p_error, aloss*mult, a_error, g_loss*self.lg, g_error, d_loss, scaled_l2_loss