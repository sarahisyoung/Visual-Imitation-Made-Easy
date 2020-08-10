import wget
import os
import tarfile

def download(url, dir_path, save_dir):
  wget.download(url, dir_path)
  print(dir_path)
  with tarfile.open(dir_path, "r") as tar_ref:
    tar_ref.extractall(save_dir)
  os.remove(dir_path)

def main():
  print('Beginning file download with wget module')
  urls = [
    ("train", "https://www.dropbox.com/s/8s7fd5ylo7979pq/train.tar.gz?dl=1"),
    ("val", 'https://www.dropbox.com/s/co65kclf8idfr04/val.tar.gz?dl=1'),
    ("test", 'https://www.dropbox.com/s/11yq82yp2l4mkz5/test.tar.gz?dl=1'),
  ]
  for dataset, link in urls:
    dir_path = "./push_" + dataset + ".zip"
    save_dir = "data/push_" + dataset
    download(link, dir_path, save_dir)




main()


