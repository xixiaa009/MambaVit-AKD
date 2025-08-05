# MambaVit-AKD:Mamba-Vision Transformer with Attention Knowledge Distillation for Missing-modality Brain Tumor Segmentation
This repository is the official PyTorch implementation of our work: Mamba-Vision Transformer with Attention Knowledge Distillation for Missing-modality Brain Tumor Segmentation.
# setup
## Environment
All our experiments are implemented based on the PyTorch framework with 45GB NVIDIA A40 GPU, and we recommend installing the following package versions:
* Python 3.10.9
* PyTorch 2.1.2
# Data preparation
* Download the preprocessed datasets BraTS2020 and BraTS2021.
* Process and store the data in NumPy array format ( .npy ) , see [RFNet](https://github.com/dyh127/RFNet/tree/main/data) for more details.
* The folder structure is assumed to be:
```plaintext
BraTS/
├── BRATS2021_Training_none_npy/
│   ├── seg
|   ├── vol
|   ├── text.txt
|   ├── train.txt
│   └── vol.txt /
├── 文件夹2/
│   ├── 文件2.ext
│   └── ...
└── README.md
```


