import torch
import numpy as np
from imageio import imread, imwrite
from generator import Generator
import glob
import os
from pathlib import Path

# ----------  ----------
def rgb_to_ycbcr(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.1687*R -0.3313*G +0.5*B + 128/255
    Cr = 0.5*R -0.4187*G -0.0813*B + 128/255
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128/255)
    G = Y - 0.34414*(Cb-128/255) -0.71414*(Cr-128/255)
    B = Y + 1.772*(Cb-128/255)
    return np.stack([R, G, B], axis=-1)

# ----------  ----------
def load_generator(checkpoint_path, device='cuda'):
    model = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)


    state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            k = k[7:]
        state_dict[k] = v


    gen_state = {}
    for k, v in state_dict.items():
        if k.startswith('generator.'):
            gen_state[k[10:]] = v
    model.load_state_dict(gen_state)
    model.eval()
    return model

# ----------  ----------
def fuse_pair(model, img1, img2, device):

    is_rgb1 = img1.ndim == 3
    is_rgb2 = img2.ndim == 3

    if is_rgb1 and is_rgb2:
        Y1, Cb1, Cr1 = rgb_to_ycbcr(img1)
        Y2, _, _ = rgb_to_ycbcr(img2)
    elif is_rgb1:
        Y1, Cb1, Cr1 = rgb_to_ycbcr(img1)
        Y2 = img2 if img2.ndim==2 else img2[:,:,0]
    elif is_rgb2:
        Y1 = img1 if img1.ndim==2 else img1[:,:,0]
        Y2, Cb1, Cr1 = rgb_to_ycbcr(img2)
    else:
        Y1, Y2 = img1, img2

    Y1_t = torch.FloatTensor(Y1).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    Y2_t = torch.FloatTensor(Y2).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        fused_Y = model(Y1_t, Y2_t).cpu().squeeze().numpy()

    if is_rgb1 or is_rgb2:
        fused = ycbcr_to_rgb(fused_Y, Cb1, Cr1)
    else:
        fused = fused_Y

    return np.clip(fused, 0, 1)

# ---------- ----------
def batch_fusion(checkpoint_path, vis_dir, ir_dir, out_dir, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)


    vis_paths = sorted(glob.glob(vis_dir))
    ir_paths  = sorted(glob.glob(ir_dir))

    if len(vis_paths) != len(ir_paths):
        print(f"Warning: Visible Images: {len(vis_paths)} ，Infrared Images: {len(ir_paths)} , The numbers don't match！")
    n = min(len(vis_paths), len(ir_paths))

    model = load_generator(checkpoint_path, device)
    print(f"Model loaded successfully, starting to merge {n}pairs of images ...")

    for i in range(n):
        vis_file = vis_paths[i]
        ir_file  = ir_paths[i]

        img_vis = imread(vis_file) / 255.0
        img_ir  = imread(ir_file) / 255.0

        fused = fuse_pair(model, img_vis, img_ir, device)


        base = os.path.splitext(os.path.basename(vis_file))[0]
        out_path = os.path.join(out_dir, f"{base}.png")
        imwrite(out_path, (fused * 255).astype(np.uint8))

        if (i+1) % 10 == 0:
            print(f"succeed {i+1}/{n}")

    print(f"All Succeed! results are saved on {out_dir}")

if __name__ == '__main__':
    # ===== actual path =====
    # checkpoint = 'checkpoints/task1/final.pth'
    # vis_pattern = './test_imgs/vis-ir/TNO/vis/*.bmp'
    # ir_pattern  = './test_imgs/vis-ir/TNO/ir/*.bmp'
    # out_dir     = './results1/vis-ir/TNO/'
    # checkpoint = 'checkpoints/task1/final.pth'
    # # vis_pattern = './test_imgs/vis-ir/RoadScene/vis/*.jpg'
    # vis_pattern = './img_RGB/vis-ir/RoadScene/*.jpg'
    # ir_pattern  = './test_imgs/vis-ir/RoadScene/ir/*.jpg'
    # out_dir     = './results2/vis-ir/RoadScene/'

    # checkpoint = 'checkpoints/task1/final.pth'
    # # vis_pattern = './test_imgs/medical/pet/*.png'
    # vis_pattern = './img_RGB/medical/*.png'
    # ir_pattern = './test_imgs/medical/mri/*.png'
    # out_dir = './results2/medical/'
    # checkpoint = 'checkpoints/task2/final.pth'
    # # vis_pattern = './test_imgs/multi-exposure/dataset1/oe/*.jpg'
    # vis_pattern = './img_RGB/multi-exposure/dataset1/*.jpg'
    # ir_pattern = './test_imgs/multi-exposure/dataset1/ue/*.jpg'
    # out_dir = './results2/multi-exposure/dataset1/'
    # checkpoint = 'checkpoints/task2/final.pth'
    # # vis_pattern = './test_imgs/multi-exposure/dataset2/oe/*.png'
    # vis_pattern = './img_RGB/multi-exposure/dataset2/*.png'
    # ir_pattern = './test_imgs/multi-exposure/dataset2/ue/*.png'
    # out_dir = './results2/multi-exposure/dataset2/'
    checkpoint = 'checkpoints/task3/final.pth'
    # vis_pattern = './test_imgs/multi-focus/far/*.jpg'
    vis_pattern = './img_RGB/multi-focus/*.jpg'
    ir_pattern = './test_imgs/multi-focus/near/*.jpg'
    out_dir = './results2/multi-focus/'
    batch_fusion(checkpoint, vis_pattern, ir_pattern, out_dir, device='cuda')
