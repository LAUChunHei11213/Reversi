import torch
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
            )
        self.fc3 = nn.Linear(512,64)

        
    def forward(self, x):
        vaild_Move = x[:,3,:,:]
        vaild_Move = vaild_Move.view(-1,64)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        #flatten
        x = x.view(-1, 128 * 8 * 8)
        
        #fc layer
        x = self.fc1(x)
        #x = self.fc2(x)
        x = self.fc3(x)
        return x*vaild_Move
class ValueCNN(nn.Module):
    def __init__(self):
        super(ValueCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tanh(x)
        return x.squeeze()
