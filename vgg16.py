# vgg16.py
import torch
import torch.nn as nn

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        # FC layers not needed for feature extraction

        if weights_path:
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict, strict=False)
            print('VGG16 weights loaded.')

    def forward(self, rgb):
        # rgb: [B,3,224,224] in [0,1]
        rgb = rgb * 255.0
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        bgr = torch.cat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], dim=1)

        h1 = torch.relu(self.conv1_1(bgr))
        h2 = torch.relu(self.conv1_2(h1))
        p1 = self.pool1(h2)

        h3 = torch.relu(self.conv2_1(p1))
        h4 = torch.relu(self.conv2_2(h3))
        p2 = self.pool2(h4)

        h5 = torch.relu(self.conv3_1(p2))
        h6 = torch.relu(self.conv3_2(h5))
        h7 = torch.relu(self.conv3_3(h6))
        p3 = self.pool3(h7)

        h8 = torch.relu(self.conv4_1(p3))
        h9 = torch.relu(self.conv4_2(h8))
        h10 = torch.relu(self.conv4_3(h9))
        p4 = self.pool4(h10)

        h11 = torch.relu(self.conv5_1(p4))
        h12 = torch.relu(self.conv5_2(h11))
        h13 = torch.relu(self.conv5_3(h12))

        return h2, h4, h7, h10, h13   
