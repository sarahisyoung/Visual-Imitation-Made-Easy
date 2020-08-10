import numpy as np
import torch
import cv2
try:
  import utils.rotation_tools as rotation_tools  # for python2
except ImportError:
  import rotation_tools #for python3
from PIL import Image
import io


def get_img_from_fig(fig, dpi=180):
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=180)
  buf.seek(0)
  img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  buf.close()
  img = cv2.imdecode(img_arr, 1)
  return img

def convert_axis_to_6d_rep(real_angles_raw, device):
  """
  Converts [x,y,z] axis representation to 6d continuous representation [a,b,c,d,e,f]
  :param real_angles_raw:
  :return: list: batch x [a,b,c,d,e,f]
  """
  real_angles = []
  for item in real_angles_raw:
    mat, jac = cv2.Rodrigues(np.array(item))
    mat = np.delete(mat, -1, 1)
    real_angles.extend(mat.reshape((1, -1)))
  return real_angles


def convert_3dmat_to_6d_rep(real_angles_raw, device):
  """
  Converts [x,y,z] axis representation to 6d continuous representation [a,b,c,d,e,f]
  :param real_angles_raw:
  :return: list: batch x [a,b,c,d,e,f]
  """
  real_angles = []
  for mat in real_angles_raw:
    mat = np.delete(mat, -1, 1)
    real_angles.extend(mat.reshape((1, -1)))
  return real_angles


def convert_6drep_to_mat(pred_raw_angles, device):
  """
          Given a batch of predicted angles in 6d representation, returns a batch of predicted angles in axis-angle representation.

          :param pred_raw_angles: batch x [a,b,c,d,e,f,g]
          :return: batch x [a,b,c]
      """
  pred_3d_angles = []
  for item in pred_raw_angles:
    mat = rotation_tools.compute_rotation_matrix_from_ortho6d(item, device)
    # raw = cv2.Rodrigues(torch.Tensor.cpu(mat.detach()).numpy()[0])
    pred_3d_angles.extend(mat)
  return pred_3d_angles



def transform_images(transform, batch_imgs, history, device):
  test_transformed_imgs = map(
    lambda x: ([transform(Image.open(i)) for i in x]),
    batch_imgs)
  test_tensored_opened_imgs = list(test_transformed_imgs)
  test_finished_imgs = list(
    map(lambda x: (
      [torch.unsqueeze(i, 0).to(device) for i in x]),
        test_tensored_opened_imgs))

  lst_of_images = []
  for i in range(history):
    img = [imgs[i] for imgs in test_finished_imgs]
    lst_of_images.append(img)

  return lst_of_images



