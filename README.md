# Visual Imitation Made Easy
### 

##### [[Project Page]](https://dhiraj100892.github.io/Visual-Imitation-Made-Easy/) [[Paper]]() [[Video]](https://youtu.be/opizQ4bXSpk)
Sarah Young, Dhiraj Gandhi, Shubham Tulsiani, Abhinav Gupta, Pieter Abbeel, Lerrel Pinto
##

Sarah Young<sup>1</sup>, Dhiraj Gandhi<sup>2</sup>, Shubham Tulsiani<sup>2</sup>, Abhinav Gupta<sup>2 3</sup>, Pieter Abbeel<sup>1</sup>, Lerrel Pinto<sup>1 4</sup>

<sup>1</sup>University of California, Berkeley, <sup>2</sup>Facebook AI Research, <sup>3</sup>Carnegie Mellon University, <sup>4</sup>New York University<br/>



<img src="images/teaser.gif" width="400">


## Usage
Below are example scripts for training and testing on provided sample data.
### Setup

1. Clone repo.
```shell
git clone https://github.com/sarahisyoung/Visual-Imitation-Made-Easy.git
```
2. Create and activate conda environment.
```shell
conda env create -f environment.yml
conda activate trashbot
```

3. Set env path.

```shell
export PYTHONPATH=$PYTHONPATH:path_to_proj/
```

### Training ###

To train with custom data, see [this](data_cleaning/README.md) for details on data processing.


1. Download the data (~45 GB) from dropbox by running this script. Feel free to comment out training/validation URL to download just a small sample of the data. 
    ```shell
    python download_data.py 
    ```

2. To train:

    ```shell
    python train.py --task push --train_dir data/train --val_dir data/val --test_dir data/test --save_dir results
    ```

### Testing


1. This command predicts on a folder of images. Output visualization is saved to specified folder.
```shell
python push_test.py --model results/exp1/policy_earlystop.pt --image_folder test_data/ --output predicted_data/
```



