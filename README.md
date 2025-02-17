# Face Recognition with GoogLeNet on VGGFace2

This repository demonstrates how to train a face recognition model using the [GoogLeNet (Inception v1)](https://arxiv.org/abs/1409.4842) architecture on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset. It provides a PyTorch-based pipeline to load the dataset, define the GoogLeNet model with auxiliary classifiers, train, and evaluate the model.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Setup](#dataset-setup)  
3. [Setup](#setup)
4. [Usage](#usage)
4. [Code Explanation](#code-explanation)  


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
        ├── vggface2_train/train  
        └── vggface2_test/test
```
3. Each of `vggface2_train/train` and `vggface2_test/test` contains multiple subfolders named like `n000001`, `n000002`, etc. Each folder contains the images for that identity.

Make sure your folder structure matches the one expected in the code, or modify the code paths accordingly.

---

## Setup

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

---

## Usage
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

---

## Code Explanation
Below is a high-level overview of the main components of the training script.

1. Imports and Hyperparameters
- import torch, import torch.nn as nn, etc.
- Set up device (CUDA or CPU), learning rate, batch size, and number of epochs.
2. Transforms
```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```
This pipeline resizes images to 256×256, center-crops to 224×224, converts to PyTorch tensors, and normalizes based on ImageNet statistics.

3. Dataset and DataLoaders

```python
train_dir = 'input/VGG-Face2/data/vggface2_train/train'
test_dir  = 'input/VGG-Face2/data/vggface2_test/test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=transform)
```
- Uses `ImageFolder` to automatically label subfolders as classes.
- Creates DataLoaders for batch loading.

4. GoogLeNet Model
```python
model = GoogLeNet(aux_logits=True, num_classes=num_classes).to(device)
```

- A custom `GoogLeNet` class that includes Inception blocks and optional auxiliary classifiers. 
- `num_classes` is set to the number of unique identities in the training set.
- `aux_logits=True` enables auxiliary heads during training.

5. Loss and Optimizer
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```
- We use `CrossEntropyLoss` for multi-class classification.
- The `Adam`optimizer with the specified learning rate.

6. Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        if model.aux_logits:
            aux1, aux2, outputs = model(inputs)
            loss = (criterion(outputs, labels)
                    + 0.5 * criterion(aux1, labels)
                    + 0.5 * criterion(aux2, labels))
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

- Feeds data to the model.
- If auxiliary classifiers are used, it combines their losses with the main output.
- Performs backpropagation and updates weights.

7. Evaluation
```python 
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

- Switches the model to evaluation mode (disabling dropout, auxiliary heads).
- Computes accuracy on the test set.

8. Plots
- The script plots training loss and accuracy over time for quick visualization.
