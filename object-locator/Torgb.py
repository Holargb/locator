import cv2
import os

input_folder = "images"
output_folder = "Atest_RGB"

os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 读取图像（包括alpha通道）
        
        # 如果图像是单通道（灰度），转换为3通道
        if len(img.shape) == 2:  # 灰度图是 (H, W)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # 如果是RGBA（4通道），去掉alpha通道
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)

print("所有图像已转换为RGB格式！")