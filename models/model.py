import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # dimezza da 224 → 112

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 → 56

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 → 28

        # Fully connected layer
        self.fc1 = nn.Linear(256 * 28 * 28, 200)  # 200 classi in TinyImageNet

    def forward(self, x):
        # B x 3 x 224 x 224
        x = self.pool1(self.conv1(x).relu())  # B x 64 x 112 x 112
        x = self.pool2(self.conv2(x).relu())  # B x 128 x 56 x 56
        x = self.pool3(self.conv3(x).relu())  # B x 256 x 28 x 28


        x = x.flatten(start_dim=1)            # B x (256*28*28)
        x = self.fc1(x)                       # B x 200
        return x