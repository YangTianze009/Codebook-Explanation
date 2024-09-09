# generate_embeddings.py
import csv
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance
from tqdm import tqdm
import sys
import torchvision.transforms.functional as TF
import ast
import matplotlib.pyplot as plt

# 添加父目录到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Visual Token contributions")
parser.add_argument('--method', type=str, default="max", help='Specify which GPU to use for computation')
parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
parser.add_argument('--model', type=int, choices=[1, 2, 3], required=True,
                    help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
args = parser.parse_args()

# 禁用梯度以节省内存
torch.set_grad_enabled(False)
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_class_mapping(class_mapping_file):
    class_mapping = {}
    with open(class_mapping_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            class_name, label = row
            class_mapping[int(label)] = class_name
    return class_mapping

def process_row(row, class_mapping):
    filename, label, contribution = row
    label = int(label)
    if label in class_mapping:
        class_name = class_mapping[label]
        # Change npy to JPEG in filename
        filename = filename.replace('.npy', '.JPEG')
        # Convert contribution string to list
        contribution_list = ast.literal_eval(contribution)
        return filename, class_name, contribution_list
    return None, None, None
            
# 预处理图像
def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x

def preprocess(img, target_image_size=256):
    s = min(img.size)
    if s < target_image_size:
        raise ValueError(f'Min dimension for image {s} < {target_image_size}')
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img



def process_token_contributions(token_contribution_file, class_mapping_file, base_image_path):
    class_mapping = load_class_mapping(class_mapping_file)

    with open(token_contribution_file, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)
        rows = list(reader)
        for row in tqdm(rows, desc="Processing rows"):
            filename, class_name, contribution_list = process_row(row, class_mapping)
            image_path = base_image_path + class_name + "/" + filename
            image = Image.open(image_path)
            processed_img = preprocess(image)
            # print("processed_img: ", processed_img.shape)
            # print("image path: ", image_path)
            # print("filename: ", filename)
            # print("classname: ", class_name)
            # print("contribution_list: ", len(contribution_list), "\n")
            contribution_list = np.array(contribution_list)
            normalized_contribution = (contribution_list - contribution_list.min()) / (contribution_list.max() - contribution_list.min())

            # 将 processed_img 转为 PIL Image 以便进行亮度调整
            processed_img_pil = Image.fromarray((processed_img.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            # 图像尺寸
            img_width, img_height = processed_img_pil.size
            patch_size = img_width // 16

            # 创建一个新的图像用于存储结果
            result_img = Image.new('RGB', (img_width, img_height))

            # 遍历每个patch并进行亮度调整
            for i in range(16):
                for j in range(16):
                    patch_index = i * 16 + j  # 计算出当前patch对应的 contribution_list 中的索引
                    brightness_factor = normalized_contribution[patch_index] ** 2
                    # 提取patch
                    left = j * patch_size
                    upper = i * patch_size
                    right = left + patch_size
                    lower = upper + patch_size
                    patch = processed_img_pil.crop((left, upper, right, lower))
                    
                    # 调整亮度，contribution为1时保持原亮度，较小时降低亮度
                    if brightness_factor < 1:
                        enhancer = ImageEnhance.Brightness(patch)
                        brightened_patch = enhancer.enhance(brightness_factor)  # 亮度降低
                    else:
                        brightened_patch = patch  # 保持原亮度
                    
                    # 将处理后的patch放回结果图像中
                    result_img.paste(brightened_patch, (left, upper))
            save_path = f"/data/ty45972/taming-transformers/codebook_explanation/results/Explanation_images/{args.method}/{class_name}"
            os.makedirs(save_path, exist_ok=True)
            result_img.save(f'{save_path}/{filename}')
            break

if __name__ == "__main__":
    token_contribution_file = f"/data/ty45972/taming-transformers/codebook_explanation/results/Explanation/ClassificationNet{args.model}_token_contributions_{args.method}.csv"
    class_mapping_file = "/data/ty45972/taming-transformers/codebook_explanation/datasets/class_mapping.csv"
    base_image_path = "/data/ty45972/taming-transformers/datasets/imagenet/val/"
    process_token_contributions(token_contribution_file, class_mapping_file, base_image_path)
