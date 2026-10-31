#  Sign Language Recognition with PyTorch

A Convolutional Neural Network (CNN) project built with PyTorch to recognize sign language alphabet gestures from RGB image data.

## 🚀 Features
- **PyTorch & CUDA:** GPU-accelerated model training and inference.
- **Dynamic Dataset Loading:** Automatically detects and maps classes using `torchvision.datasets.ImageFolder`.
- **Custom CNN Architecture:** Optimized sequential structure utilizing Conv2d, ReLU, MaxPool2d, and Linear layers.
- **Inference Script (`test.py`):** Loads trained weights (`.pth`) and predicts sign language letters from custom images.

##  Project Structure
```text
├── Datasets/
│   └── train/
│       └── alphabe/       # Class folders (A, B, C...) [Excluded via .gitignore]
├── sign_language.pth      # Trained model weights [Excluded via .gitignore]
├── train.py               # Model training script
└── test.py                # Model test & prediction script
