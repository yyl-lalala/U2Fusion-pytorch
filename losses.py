# losses.py
import torch
import torch.nn.functional as F
import numpy as np

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g = g.unsqueeze(0) * g.unsqueeze(1)  # [size, size]
    g = g.view(1, 1, size, size)
    return g

def ssim_loss(img1, img2, window_size=11, sigma=1.5):
    window = gaussian_window(window_size, sigma).to(img1.device)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(img1, window, padding=0)
    mu2 = F.conv2d(img2, window, padding=0)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # 返回每个样本的SSIM值 [B]
    return ssim_map.mean(dim=[1, 2, 3])

def fro_loss(img1, img2):
    # 每个样本的MSE（除以像素数）
    return ((img1 - img2) ** 2).mean(dim=[1, 2, 3])