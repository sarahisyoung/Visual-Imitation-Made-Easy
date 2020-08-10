import glob
import json
import os
import time

import torch

from models.forward_model import TrashPolicy
from scripts.push_trainer import PushTrainer
from utils.get_result_vids import makeVideo


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_dir', type=str)
  parser.add_argument('--val_dir', type=str)
  parser.add_argument('--test_dir', type=str)
  parser.add_argument('--which_gpu', '-gpu_id', default=0)
  parser.add_argument('--save_dir', type=str, default="results")
  parser.add_argument('--exp_name', type=str, default='todo')
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--pretrained', type=int, default=0)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--net_type', default='alex')
  parser.add_argument('--feat_dim', type=int, default=256)
  parser.add_argument('--epochs', type=int, default=60)
  parser.add_argument('--model', type=str, default="policy", choices=["policy"])
  parser.add_argument('--env', type=str, default="trash")
  parser.add_argument('--history', type=int, default=1)
  parser.add_argument('--mult', type=int, default=0)
  parser.add_argument('--mirror', type=int, default=0)
  parser.add_argument('--rot', type=str, default="6d-d")
  parser.add_argument('--data_size', '-fraction of runs to use', type=float, default=1)
  parser.add_argument('--l1', type=float, default=0)  # weight for l1 loss
  parser.add_argument('--l2', type=float, default=1)  # weight for l2 loss
  parser.add_argument('--l3', type=float, default=0)  # weight for direction loss
  parser.add_argument('--lg', type=float, default=0)  # weight for gripper_loss
  parser.add_argument('--rad', type=str, default="None")
  parser.add_argument('--xyz_normed', type=int, default=0)
  parser.add_argument('--task', type=str, required=True, choices=["push"])
  parser.add_argument('--train', type=int, default=1)
  parser.add_argument('--grip_file', type=str, required=False)
  parser.add_argument('--rand', type=int, default=0)

  args = parser.parse_args()

  # convert to dictionary
  params = vars(args)

  ##################################
  ### CREATE DIRECTORY FOR LOGGING
  ##################################

  logdir_prefix = 'trashpolicy_' + args.exp_name + "_" + args.net_type

  data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.save_dir)

  if not (os.path.exists(data_path)):
    os.makedirs(data_path)

  logdir = logdir_prefix + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
  logdir = os.path.join(data_path, logdir)
  params['logdir'] = logdir
  if not (os.path.exists(logdir)):
    os.makedirs(logdir)
    os.makedirs(logdir + "/valimages/")
    os.makedirs(logdir + "/trainimages/")
    os.makedirs(logdir + "/testimages/")

  print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

  device = torch.device('cuda:' + str(params['which_gpu']) if torch.cuda.is_available() else "cpu")

  print("\n\n\nTraining a " + params['model'] + "\n\n\n")

  model = TrashPolicy(params).to(device)

  with open(logdir + '/params.json', 'w') as outfile:
    json.dump(params, outfile, indent=4, separators=(',', ': '), sort_keys=True)
    # add trailing newline for POSIX compatibility
    outfile.write('\n')

  if args.task == "push":
    trainer = PushTrainer(
      model,
      params
    )
  elif args.task == "stack":
    trainer = StackTrainer(
      model,
      params
    )
  trainer.train(logdir)

  makeVideo(logdir, logdir, "test")
  makeVideo(logdir, logdir, "val")
  makeVideo(logdir, logdir, "train")

  vids = sorted(glob.glob(logdir + "/*.avi"))
  for ff in vids:
    os.system("ffmpeg -i " + ff + " " + ff[:-4] + ".mp4")
    os.remove(ff)


if __name__ == "__main__":
  main()
