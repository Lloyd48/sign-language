import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import os
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")
    train_data = torchvision.datasets.ImageFolder(root="Datasets/train",transform=transform)
    print(train_data.classes)
    model = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.Flatten(),nn.Linear(64*16*16,27)).to(device)
    model.load_state_dict(torch.load("sign_language.pth",weights_only=True,map_location=device))
    Image = Image.open("C:/Users/metin/Downloads/archive/Test_Alphabet/Blank/1d85acd0-22ec-43a7-aecd-d1369a770a2c.rgb_0000.png").convert("RGB")
    transformed_Image = transform(Image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        guess = model(transformed_Image)
        
        index = torch.argmax(guess,dim=1).item()
        print(train_data.classes[index])