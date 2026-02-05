#  PhysioNet - Digitization of ECG Images - 6th Place Solution

This repository contains the 6th place solution for the PhysioNet - Digitization of ECG Images competition. For a detailed discussion, please refer to the [competition discussion thread](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/6th-place-solution).

## Environment

This project was developed using the following hardware and software environment:

### Workstation Specifications

- **CPU**: Intel(R) Xeon(R) Gold 5218R @ 2.10GHz
- **Memory**: 256 GB RAM
- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Operating System**: Ubuntu 20.04.6 (Kernel 5.4.0-181-generic)

## Summary

In this competition, the objective is to digitize 12-lead ECG signals from diverse physical formats, including hard-copy printouts, scans, and photos. The proposed solution follows a robust three-stage pipeline:
- Stage 0: `Image Geometric Normalization`. The model predicts spatial keypoints to execute a homography transformation. This corrects rotation and perspective distortions, producing standardized, aligned images for downstream tasks.
- Stage 1: `Grid-based Image Rectification`. A dense grid of coordinates and structural gridlines is predicted to perform non-linear warping. This "unrolls" and flattens the normalized images, effectively eliminating local surface distortions.
- Stage 2: `ECG Signal Digitization`. Utilizing a UNet-based architecture, the model performs direct signal regression to convert rectified visual crops into 1D numerical time-series (millivolt values) for each lead.

I adopted the strong baseline provided by [hengck23](https://www.kaggle.com/code/hengck23/demo-submission)  for Stages 0 and 1 and the primary contribution focuses on the optimization of Stage 2. 

## Setup

1. Create a new conda environment:
```bash
conda create -n PhysioNet python=3.10 -y
conda activate PhysioNet
```

2. Clone the repository:
```bash
git clone https://github.com/GWwangshuo/Kaggle-2025-PhysioNet.git
cd Kaggle-2025-PhysioNet
````

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

#### 1. Data Preparation
Run the following command to download the official competition dataset and construct the customized datasets used by the Stage 2 signal regression model. As an alternative, the Stage 2 datasets can also be generated using [hengck23's code](https://www.kaggle.com/code/hengck23/demo-submission).
```bash
python3 src/tools/prepare_data.py
```

#### 2. Model Training
Launch the training pipeline for the signal regression model:

- **MONAI U-Net (1 residual unit, spatial size: 128x256x256, 7 classes)**
  ```bash
  torchrun --nproc_per_node=8 src/train.py
  ```


#### 3. Model Prediction
1. Download the pretrained checkpoints from [this link](https://www.kaggle.com/datasets/sjtuwangshuo/gavin-submit-physionet). If you have trained the model yourself, replace the existing weights with your custom-trained ones.
2. Update the dataset path `LIBS_DIR` and the absolute checkpoint path `weights` in the `submit.ipynb` notebook.
3. Excute the `submit.ipynb` notebook.

##### Model Performance

| No   | Architecture                | Parameters | Epochs | Extra Data | Resampling Length | TTA   | Public LB | Private LB |
| ---- | --------------------------- | ---------- | ------ | ---------- | ----------------- | ----- | --------- | ---------- |
| 1    | UNet with Queries           | 21.77M     | 100    | FALSE      | 5120              | HFLIP | 22.15119  | 21.91432   |
| 2    | UNet                        | 21.77M     | 150    | FALSE      | 5120              | HFLIP | 22.43446  | 22.2459    |
| 3    | UNet                        | 21.77M     | 80     | TRUE       | 5120              | HFLIP | 22.24785  | 22.0343    |
| 4    | UNet                        | 21.77M     | 70     | FALSE      | 10250             | HFLIP | 21.75206  | 21.60695   |
| 5    | UNet                        | 21.77M     | 120    | FALSE      | 10250             | HFLIP | 21.66284  | 21.61265   |
| 6    | UNet                        | 21.77M     | 100    | FALSE      | 5120              | HFLIP | 22.40282  | 22.20383   |
| 7    | UNet with ResNet 50 Encoder | 24.42M     | 140    | FALSE      | 5120              | HFLIP | 21.66284  | 21.61265   |



##### Acknowledgments

We would like to thank [hengck23](https://www.kaggle.com/code/hengck23/demo-submission) for sharing his excellent work, which provided valuable reference of the overall pipelines.
