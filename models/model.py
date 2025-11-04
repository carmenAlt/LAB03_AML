import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # Fully connected layer
        self.fc1 = nn.Linear(256 * 8 * 8, 200)  # 200 classi Tiny-ImageNet

    def forward(self, x):
        x = self.pool1(self.conv1(x).relu())
        x = self.pool2(self.conv2(x).relu())
        x = self.pool3(self.conv3(x).relu())

        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x
