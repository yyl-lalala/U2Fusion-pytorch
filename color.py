import os
import cv2
import numpy as np

#example
rgb_folder = './img_RGB/vis-ir/RoadScene/'
intensity_folder = './results/vis-ir/RoadScene/'
output_folder = './fused1/vis-ir/RoadScene/'


os.makedirs(output_folder, exist_ok=True)


rgb_images = sorted(os.listdir(rgb_folder))
intensity_images = sorted(os.listdir(intensity_folder))


for rgb_image, intensity_image in zip(rgb_images, intensity_images):
    rgb_path = os.path.join(rgb_folder, rgb_image)
    img_rgb = cv2.imread(rgb_path)
    if img_rgb is None:
        print(f"Can't read the image: {rgb_image}")
        continue

    # RGB to YCrCb
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycrcb)


    intensity_path = os.path.join(intensity_folder, intensity_image)
    img_y = cv2.imread(intensity_path, cv2.IMREAD_GRAYSCALE)
    if img_y is None or img_y.shape != Y.shape:
        print(f"Intensity image error or size mismatch: {intensity_image}")
        continue

    merged_ycrcb = cv2.merge([img_y, Cr, Cb])

    # YCrCb to RGB
    img_fusion_rgb = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)


    output_path = os.path.join(output_folder, rgb_image)
    cv2.imwrite(output_path, img_fusion_rgb)
    print(f"Images are saved: {output_path}")

print("Image fusion complete！")
