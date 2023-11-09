# PinNet: Visually Salient 3D Keypoint Detection in Implicit Scene Representation
This repo is the implementation of the paper 'PinNet: Visually Salient 3D Keypoint Detection in Implicit Scene Representation'

## Introduction
Semantically meaningful keypoints in 3D scenes are essential for scene understanding and analysis tasks. When dealing with computer vision-related downstream tasks such as Camera Pose Estimation or Visual Simultaneous Localization and Mapping (vSLAM), it is beneficial to locate these 3D keypoints at visually salient positions. This allows them to be correlated with their 2D counterparts in observations using techniques like feature matching. LiDAR point clouds serve as an effective representation of the 3D scene, offering a cost-efficient and accurate solution. However, point clouds solely encode 3D geometric information and lack visual appearance details. The discrepancy between these two domains leads difficulties in locating visually salient keypoints in such scenes. Moreover, the sparsity of point clouds leads to inadequate scanning of the scene, resulting in the exclusion of semantically meaningful points from the point cloud, which poses challenges for the precise detection of 3D keypoints. To address these issues, we propose PinNet, a novel model designed to accurately detect visually salient keypoints in 3D scenes represented by point clouds. Through extensive experiments, we demonstrate the effectiveness of our method, affirming its value for various downstream applications.

![alt text](https://github.com/hansunuts/pinnet/blob/main/paper/intro_kp.png?raw=true)
![alt text](https://github.com/hansunuts/pinnet/blob/main/paper/fig_model.png?raw=true)

## Quick Start
### Installation
```
git clone https://github.com/hansunuts/pinnet.git
cd pinnet

conda env create -f environment.yml
conda activate pinnet
```

### Kp3D Dataset
#### Download Kp3D Dataset
Coming soon.
#### Sample Data
You can find some sample data under \sample_data

### Download Pretrained Model
1. Download pretrained model from https://drive.google.com/drive/folders/1rnZzZ1lO5lO-R0kCxpuRnCl0LhXLv3b1?usp=sharing

2. Copy pretrained model to corresponding experiment folder. Eg. copy plane.pth to experiments/plane

Currently only pretrained model for plane category is available. More pretrained model will be available soon.

### Run keypoint detection examples
```
python sandbox/vis_kps3d.py
```
