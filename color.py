import os
import cv2
import numpy as np

rgb_folder = './img_RGB/vis-ir/RoadScene/'  # 第一个文件夹RGB图像路径
intensity_folder = './results1/vis-ir/RoadScene/'  # 第二个文件夹强度图像路径
output_folder = './fused1/vis-ir/RoadScene/'  # 第三个文件夹输出路径


os.makedirs(output_folder, exist_ok=True)


rgb_images = sorted(os.listdir(rgb_folder))
intensity_images = sorted(os.listdir(intensity_folder))


for rgb_image, intensity_image in zip(rgb_images, intensity_images):
    # 读取RGB图像
    rgb_path = os.path.join(rgb_folder, rgb_image)
    img_rgb = cv2.imread(rgb_path)
    if img_rgb is None:
        print(f"无法读取图像: {rgb_image}")
        continue

    # RGB转YCrCb
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycrcb)

    # 读取强度图像
    intensity_path = os.path.join(intensity_folder, intensity_image)
    img_y = cv2.imread(intensity_path, cv2.IMREAD_GRAYSCALE)
    if img_y is None or img_y.shape != Y.shape:
        print(f"强度图像错误或尺寸不匹配: {intensity_image}")
        continue

    merged_ycrcb = cv2.merge([img_y, Cr, Cb])

    # YCrCb转回RGB
    img_fusion_rgb = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)


    output_path = os.path.join(output_folder, rgb_image)
    cv2.imwrite(output_path, img_fusion_rgb)
    print(f"保存图像: {output_path}")

print("图像融合完成！")
