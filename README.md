# Pinnet

## Setup anaconda environment
conda env create -f environment.yml
conda activate pinnet

## Run keypoint detection examples
Download pretrained model from https://drive.google.com/drive/folders/1rnZzZ1lO5lO-R0kCxpuRnCl0LhXLv3b1?usp=sharing
Currently only pretrained model for plane category is available. More pretrained model will be available soon.

python sandbox/vis_kps3d.py
