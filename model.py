# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg16 import Vgg16
from losses import ssim_loss, fro_loss
from generator import Generator

def features_grad(features):
    kernel = torch.tensor([[1/8, 1/8, 1/8],
                           [1/8, -1, 1/8],
                           [1/8, 1/8, 1/8]]).float().to(features.device)
    kernel = kernel.view(1, 1, 3, 3)
    # 对每个通道做卷积
    B, C, H, W = features.shape
    grad = F.conv2d(features.reshape(-1, 1, H, W), kernel, padding=1).view(B, C, H, W)
    return grad

class FusionModel(nn.Module):
    def __init__(self, vgg_weights_path=None):
        super().__init__()
        self.generator = Generator()
        self.vgg = Vgg16(vgg_weights_path) if vgg_weights_path else None
        if self.vgg:
            self.vgg.eval()
            for p in self.vgg.parameters():
                p.requires_grad = False

    def forward(self, S1, S2):
        return self.generator(S1, S2)

    def content_loss(self, S1, S2, fused, c):
        # S1, S2: [B,1,H,W]
        ssim1 = 1 - ssim_loss(S1, fused)
        ssim2 = 1 - ssim_loss(S2, fused)
        mse1 = fro_loss(fused, S1)
        mse2 = fro_loss(fused, S2)

        if self.vgg is not None:
            # 重采样到224x224并转换为3通道
            s1_3 = S1.expand(-1, 3, -1, -1)
            s2_3 = S2.expand(-1, 3, -1, -1)
            s1_224 = F.interpolate(s1_3, size=(224,224), mode='nearest')
            s2_224 = F.interpolate(s2_3, size=(224,224), mode='nearest')

            with torch.no_grad():
                feats1 = self.vgg(s1_224)
                feats2 = self.vgg(s2_224)

            ws1, ws2 = [], []
            for f1, f2 in zip(feats1, feats2):
                m1 = (features_grad(f1) ** 2).mean(dim=[1,2,3])
                m2 = (features_grad(f2) ** 2).mean(dim=[1,2,3])
                ws1.append(m1)
                ws2.append(m2)
            ws1 = torch.stack(ws1, dim=1).mean(dim=1) / c   # [B]
            ws2 = torch.stack(ws2, dim=1).mean(dim=1) / c
            weights = torch.softmax(torch.stack([ws1, ws2], dim=1), dim=1)  # [B,2]
        else:
            # 如果没有VGG，使用相等权重（退化情况）
            weights = torch.ones(S1.size(0), 2).to(S1.device) / 2

        w1, w2 = weights[:, 0], weights[:, 1]
        loss_ssim = (w1 * ssim1 + w2 * ssim2).mean()
        loss_mse = (w1 * mse1 + w2 * mse2).mean()
        return loss_ssim + 20 * loss_mse