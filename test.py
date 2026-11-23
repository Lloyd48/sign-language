import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import os
import cv2
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    
    TEST_PATH = BASE_DIR / "Datasets" / "test"
    def get_data_loaders():
        test_transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=test_transform)
        print("Test sınıfları:", test_data.class_to_idx)
        test_loader = DataLoader(test_data, batch_size=128,num_workers=6,pin_memory=True, shuffle=False)
        return test_loader,test_data,test_transform
    test_loader,test_data,transform = get_data_loaders()
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
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_calculation = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load("sign_language_model2.pth", weights_only=True))
    model.eval()
    kamera = cv2.VideoCapture(0)
    while True:
        basarı,kare = kamera.read()
        if not basarı:
            print("Kamerada görüntü alınamadı")
            break
        rgb_kare = cv2.cvtColor(kare,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_kare)
        dönüştürülmüş = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
                outputs = model(dönüştürülmüş)
                index = torch.argmax(outputs, dim=1).item()
                tahmin = test_data.classes[index]
        cv2.putText(kare,tahmin,(50,90),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2.0,(0,0,255),1)
        cv2.imshow("Sign_Language",kare)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    kamera.release()
    cv2.destroyAllWindows()
