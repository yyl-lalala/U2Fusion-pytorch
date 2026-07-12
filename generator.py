# generator.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n=44):
        super().__init__()
        self.conv1 = nn.Conv2d(2, n, 3, padding=1, padding_mode='reflect')
        self.dense1 = nn.Conv2d(n, n, 3, padding=1, padding_mode='reflect')
        self.dense2 = nn.Conv2d(n*2, n, 3, padding=1, padding_mode='reflect')
        self.dense3 = nn.Conv2d(n*3, n, 3, padding=1, padding_mode='reflect')
        self.dense4 = nn.Conv2d(n*4, n, 3, padding=1, padding_mode='reflect')
        self.dense5 = nn.Conv2d(n*5, n, 3, padding=1, padding_mode='reflect')
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x: [B,2,H,W]
        out = self.leaky_relu(self.conv1(x))  # B,44,H,W
        features = [out]  # 收集所有层输出

        # dense1
        out1 = self.leaky_relu(self.dense1(out))
        features.append(out1)
        concat2 = torch.cat(features, dim=1)  # B,88,H,W

        out2 = self.leaky_relu(self.dense2(concat2))
        features.append(out2)
        concat3 = torch.cat(features, dim=1)  # B,132,H,W

        out3 = self.leaky_relu(self.dense3(concat3))
        features.append(out3)
        concat4 = torch.cat(features, dim=1)  # B,176,H,W

        out4 = self.leaky_relu(self.dense4(concat4))
        features.append(out4)
        concat5 = torch.cat(features, dim=1)  # B,220,H,W

        out5 = self.leaky_relu(self.dense5(concat5))
        features.append(out5)

        return torch.cat(features, dim=1)  # B,264,H,W → 6*44

class Decoder(nn.Module):
    def __init__(self, n=44):
        super().__init__()
        self.conv1 = nn.Conv2d(n*6, 128, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1, padding_mode='reflect')
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.leaky_relu(self.conv1(x))
        out = self.leaky_relu(self.conv2(out))
        out = self.leaky_relu(self.conv3(out))
        out = self.tanh(self.conv4(out)) / 2 + 0.5
        return out

class Generator(nn.Module):
    def __init__(self, n=44):
        super().__init__()
        self.encoder = Encoder(n=n)
        self.decoder = Decoder(n=n)

    def forward(self, I1, I2):
        x = torch.cat([I1, I2], dim=1)  # [B,2,H,W]
        code = self.encoder(x)
        out = self.decoder(code)
        return out