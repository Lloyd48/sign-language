import torch 
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import os
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Current Device: {device}")
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
    model = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.Flatten(),nn.Linear(64*16*16,27)).to(device)
    train_data = torchvision.datasets.ImageFolder(root="Datasets/train",transform=transform)
    print(train_data.classes)
    train_load = DataLoader(train_data,shuffle=True,batch_size=256,num_workers=4,pin_memory=True)
    hata_hesaplayıcı = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0007)
    for epoch in range(15):
        print(epoch)
        for image,label in train_load:
            image = image.to(device)
            label = label.to(device)
            guess = model(image)
            loss = hata_hesaplayıcı(guess,label)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Model created")
    torch.save(model.state_dict(),"sign_language.pth")
    print("File saved succesfully")