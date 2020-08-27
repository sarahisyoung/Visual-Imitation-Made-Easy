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
    ("train", "https://www.dropbox.com/s/9eojzne9a0jjlqx/pabti_push_train.tar.gz?dl=1"),
    ("val", 'https://www.dropbox.com/s/c2xfig00vl9j43k/pabti_push_val.tar.gz?dl=1'),
    ("test", 'https://www.dropbox.com/s/be1rrprn9lws75o/pabti_push_test.tar.gz?dl=1'),
  ]
  for dataset, link in urls:
    dir_path = "./push_" + dataset + ".zip"
    save_dir = "data/push_" + dataset
    download(link, dir_path, save_dir)




main()


