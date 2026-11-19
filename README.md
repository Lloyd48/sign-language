# Sign Language Recognition with CNN

A deep learning project that recognizes hand gestures using a custom Convolutional Neural Network (CNN) built with PyTorch.

The model is trained to classify 26 different hand gesture classes representing the letters of the alphabet.

## Model Architecture

This project uses a custom ResNet-inspired CNN architecture with residual connections.

The network consists of:

- **4 Residual Blocks**
- Convolutional layers with 64, 128, and 256 channels
- Batch Normalization
- ReLU activation
- Max Pooling
- Adaptive Average Pooling
- Dropout (0.2)
- Fully Connected Classification Layer (26 classes)

Residual connections help the network learn deeper features and improve training stability.

## Model Performance

The model was trained for 30 epochs and evaluated on a separate test dataset.

| Metric | Result |
|---|---:|
| Training Epochs | 30 |
| Test Accuracy | **98.98%** |
| Number of Classes | 26 |
| Input Image Size | 128 × 128 |
| Optimizer | Adam |
| Loss Function | Cross Entropy Loss |
| Learning Rate Scheduler | StepLR |

The training and test datasets were kept separate during the process. The test dataset was not used for training.

## Dataset

The dataset contains images of hand gestures organized into 26 class folders.

The expected directory structure is:

```text
Datasets/
├── train/
│   ├── a/
│   ├── b/
│   ├── c/
│   └── ...
└── test/
    ├── a/
    ├── b/
    ├── c/
    └── ...
```

**Why is the dataset not included in this repository?**

The dataset is not uploaded because of its large size and to keep the repository lightweight. Users need to provide their own dataset following the directory structure above.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Pillow

Install the required libraries:

```bash
pip install torch torchvision pillow
```

## Training

1. Place your dataset inside the `Datasets` directory.
2. Make sure the training images are organized into class folders.
3. Run the training script:

```bash
python train.py
```

After training, the model weights are saved as:

```text
sign_language_model.pth
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- Convolutional Neural Networks (CNN)
- Deep Learning

## License

This project is licensed under the MIT License.
