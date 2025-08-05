# MambaVit-AKD:Mamba-Vision Transformer with Attention Knowledge Distillation for Missing-modality Brain Tumor Segmentation
This repository is the official PyTorch implementation of our work: Mamba-Vision Transformer with Attention Knowledge Distillation for Missing-modality Brain Tumor Segmentation.
# setup
## Environment
All our experiments are implemented based on the PyTorch framework with 45GB NVIDIA A40 GPU, and we recommend installing the following package versions:
* Python 3.10.9
* PyTorch 2.1.2
# Data preparation
* Download the preprocessed datasets BraTS2020 and BraTS2021 from [kaggle](https://www.kaggle.com/datasets).
* Process and store the data in NumPy array format ( .npy ) , see [RFNet](https://github.com/dyh127/RFNet/tree/main/data) for more details.
* The folder structure is assumed to be:
```plaintext
data/
├── BRATS2020_data/
│   ├── seg
│   │   ├── BraTS20_Training_001_seg.npy
│   │   ├── BraTS20_Training_002_seg.npy
│   │   ├── BraTS20_Training_003_seg.npy
│   │   ├──......
|   ├── vol
│   │   ├──BraTS20_Training_001_vol.npy
│   │   ├──BraTS20_Training_002_vol.npy
│   │   ├──BraTS20_Training_003_vol.npy
│   │   ├──......
|   ├── text.txt
|   ├── train.txt
│   └── vol.txt /
├── BRATS2021_data/
│   ├── seg
│   ├── ....../
│   ├── vol
│   ├── ....../
│   └── ......
└── ......
```
# Running Experiments


