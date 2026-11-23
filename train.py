import torch;import torch.nn as nn;import torchvision;import torchvision.transforms as transforms;import torch.optim as optim;import os;from pathlib import Path;from PIL import Image;
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    TRAIN_PATH = BASE_DIR / "Datasets" / "train"
    print("Python dosyası:", Path(__file__).resolve())
    print("Aranan klasör:", TRAIN_PATH)
    print("Klasör mevcut mu:", TRAIN_PATH.exists())
    def get_data_loaders():
        train_transform = transforms.Compose([transforms.Resize((128, 128)),transforms.RandomAffine(degrees=8,translate=(0.08, 0.08),scale=(0.9, 1.1),shear=5),transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.15,hue=0.03),transforms.RandomPerspective(distortion_scale=0.15,p=0.2),transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=train_transform)
        print("Train sınıfları:", train_data.class_to_idx)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True,num_workers=8, pin_memory=True)
        return train_loader
    train_loader = get_data_loaders()
    class Cnn(nn.Module):
        def __init__(self):
            super(Cnn, self).__init__()
            self.shortcut1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))
            self.block1 = nn.Sequential(nn.Conv2d(3,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64))

            self.shortcut2 = nn.Sequential(nn.Conv2d(64,128,1,1,0),nn.BatchNorm2d(128))
            self.block2 = nn.Sequential(nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128))

            self.shortcut3 = nn.Sequential(nn.Conv2d(128,256,1,1,0),nn.BatchNorm2d(256))
            self.block3 = nn.Sequential(nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256,256,3,1,1),nn.BatchNorm2d(256))
            
            self.shortcut4 = nn.Sequential(nn.Conv2d(256,256,1,1,0),nn.BatchNorm2d(256))
            self.block4 = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256,256,3,1,1),nn.BatchNorm2d(256))
            self.block5 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),)
            self.shortcut5 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1),nn.BatchNorm2d(256),)   
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.BatchNorm1d(256),nn.Dropout(0.2),nn.Linear(256, 26))
        def forward(self, x):
            identity = self.shortcut1(x)
            out = self.block1(x)
            out += identity
            out = self.relu(out)
            out = self.pool(out)

            identity = self.shortcut2(out)
            out = self.block2(out)
            out += identity
            out = self.relu(out)
            out = self.pool(out)
            identity = self.shortcut3(out)
            out = self.block3(out)
            out += identity
            out = self.relu(out)
            out = self.pool(out)
            identity = self.shortcut4(out)
            out = self.block4(out)
            out += identity
            out = self.relu(out)
            identity = self.shortcut5(out)
            
            out = self.block5(out)
            out += identity
            out = self.relu(out)

            out = self.classifier(out)
            return out
    model = Cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_calculation = nn.CrossEntropyLoss()
    def train(model, device, train_loader, optimizer,loss_calculation, epochs=1):
        model.train()
        for epoch in range(epochs):
            loss_value = 0
            avg_loss = 0
            correct = 0
            total = 0
            for image,label in train_loader:
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(image)
                loss = loss_calculation(output, label)
                loss.backward()
                optimizer.step()
                _,index = torch.max(output, dim=1)
                total += label.size(0)
                correct += (index == label).sum().item()
                loss_value += loss.item()
            scheduler.step()
            avg_loss = loss_value / len(train_loader)
            print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
    train(model, device, train_loader, optimizer, loss_calculation, epochs=30)
    torch.save(model.state_dict(), "sign_language_model.pth")
