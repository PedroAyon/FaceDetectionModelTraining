# Face Recognition with GoogLeNet on VGGFace2

This repository demonstrates how to train a face recognition model using the [GoogLeNet (Inception v1)](https://arxiv.org/abs/1409.4842) architecture on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset. It provides a PyTorch-based pipeline to load the dataset, define the GoogLeNet model with auxiliary classifiers, train, and evaluate the model.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Setup](#dataset-setup)  
3. [Installation](#installation)  
4[Code Explanation](#code-explanation)  


---

## Project Overview

Face recognition is a classic task in computer vision. In this project, we use the VGGFace2 dataset, which contains face images from a large variety of individuals, to train a GoogLeNet-based model. 

### Key Features
- **GoogLeNet Architecture**: Includes multiple inception blocks and optional auxiliary classifiers for improved gradient flow during training.
- **PyTorch Implementation**: The model, data loading, and training pipeline are implemented in PyTorch.
- **ImageFolder**: We use the `torchvision.datasets.ImageFolder` class to organize the dataset by identities (one folder per identity).
- **Basic Data Augmentation**: Includes resizing, center cropping, and normalization. Additional augmentations can be added as needed.

---

## Dataset Setup

1. **Download the VGGFace2 dataset** from the official website or another source.  
2. **Extract** the dataset so that you have:
```
input/  
└── VGG-Face2/  
    ├── data/  
    │   ├── vggface2_train/train  
    │   └── vggface2_test/test
```
3. Each of `vggface2_train/train` and `vggface2_test/test` contains multiple subfolders named like `n000001`, `n000002`, etc. Each folder contains the images for that identity.

Make sure your folder structure matches the one expected in the code, or modify the code paths accordingly.

---

## Installation

1. **Clone** this repository:
```bash
git clone https://github.com/PedroAyon/FaceDetectionModelTraining.git
```
2. Install **Python3.10** if you don’t already have it.
3. Create a virtual environment (recommended):
```bash
python3.10 -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows 
```
4. Install requirements:
```bash
pip install -r requirements.txt
```

# Usage
1. Adjust hyperparameters as desired in the code (e.g., learning rate, batch size, number of epochs).
2. Run the training script:
```bash
python train_vggface2_googlenet.py 
```
3. Monitor the training output in your console. The script will print:
- Epoch number
- Mini-batch loss
- Training accuracy per epoch
4. Check the plots: After training, a window (or inline plot if you’re in a notebook) will show training loss and accuracy curves.
5. Evaluation: The code also evaluates the model on the test set and prints the final test accuracy.


