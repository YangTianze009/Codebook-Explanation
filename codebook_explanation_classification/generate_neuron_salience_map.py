import csv
import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm
import sys
import torchvision.transforms.functional as TF
import ast
import argparse

# 添加父目录到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 禁用梯度以节省内存
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

parser = argparse.ArgumentParser(description="Visual Token contributions")
parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
parser.add_argument('--model', type=int, choices=[1, 2, 3], required=True,
                    help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
parser.add_argument('--labels', type=int, nargs='+', default=[7, 8, 9, 10, 11, 12, 15, 82, 161, 175, 715], required=False, help='Specify one or more labels for image selection')
parser.add_argument('--num_images', type=int, default=50, help='Specify the number of images to select for processing')
parser.add_argument('--visualization_mode', type=str, choices=['brightness', 'highlight'], default='highlight',
                    help="Choose the visualization mode: 'brightness' for adjusting brightness based on contribution, 'highlight' for highlighting only the highest contribution area")
args = parser.parse_args()

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

def load_and_select_images(token_contribution_file, label, num_images=100):
    selected_images = []
    with open(token_contribution_file, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        rows = [row for row in reader if int(row[1]) == label]
        
        # 如果可用图像少于请求的数量，使用所有可用图像
        actual_num_images = min(len(rows), num_images)
        if len(rows) > 0:
            random.shuffle(rows)
            selected_images = rows[:actual_num_images]
        
        # 如果没有找到图像，返回空列表
    
    return selected_images

def process_and_save_images(selected_images, layer_folder, save_root):
    layer_name = os.path.basename(layer_folder)  # 获取当前层的名称（子文件夹名）
    csv_files = [f for f in os.listdir(layer_folder) if f.endswith('.csv')]
    with tqdm(total=len(csv_files), desc=f"Processing {layer_name}", unit="csv file") as pbar:
        for csv_file in csv_files:
            csv_path = os.path.join(layer_folder, csv_file)
            channel_name = os.path.splitext(csv_file)[0]

            with open(csv_path, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                rows = list(reader)

                processed_any_image = False
                for row in selected_images:
                    filename, label, _ = row
                    filename_base = os.path.basename(filename)
                    for row in rows:
                        if row[0] == filename_base:
                            contribution_list = ast.literal_eval(row[2])
                            contribution_array = np.array(contribution_list)
                            max_activation = contribution_array.max()

                            if max_activation < 1e-3:
                                continue

                            normalized_contribution = (contribution_array - contribution_array.min()) / (contribution_array.max() - contribution_array.min())

                            image_folder = filename.split('_')[0]
                            image_name = filename.split('_')[1].replace('.npy', '.png')
                            image_path = os.path.join(f"/data/ty45972/taming-transformers/datasets/imagenet_VQGAN_generated/{image_folder}/", image_name)
                            image = Image.open(image_path)
                            processed_img = preprocess(image)

                            processed_img_pil = Image.fromarray((processed_img.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))

                            img_width, img_height = processed_img_pil.size
                            patch_size = img_width // 16

                            result_img = processed_img_pil.copy()
                            draw = ImageDraw.Draw(result_img)

                            if args.visualization_mode == 'brightness':
                                for i in range(16):
                                    for j in range(16):
                                        patch_index = i * 16 + j
                                        brightness_factor = normalized_contribution[patch_index] ** 2

                                        left = j * patch_size
                                        upper = i * patch_size
                                        right = left + patch_size
                                        lower = upper + patch_size
                                        patch = processed_img_pil.crop((left, upper, right, lower))

                                        if brightness_factor < 1:
                                            enhancer = ImageEnhance.Brightness(patch)
                                            brightened_patch = enhancer.enhance(brightness_factor)
                                        else:
                                            brightened_patch = patch

                                        result_img.paste(brightened_patch, (left, upper))

                            elif args.visualization_mode == 'highlight':
                                max_value = normalized_contribution.max()
                                max_indices = np.where(normalized_contribution == max_value)[0]  # 获取所有最大值的索引

                                # 找到所有最大值的区域，并计算这些区域的外部边界
                                min_row, min_col = 16, 16
                                max_row, max_col = 0, 0
                                for max_index in max_indices:
                                    i, j = divmod(max_index, 16)
                                    min_row = min(min_row, i)
                                    min_col = min(min_col, j)
                                    max_row = max(max_row, i)
                                    max_col = max(max_col, j)

                                # 计算外部边界的像素位置
                                left = min_col * patch_size
                                upper = min_row * patch_size
                                right = (max_col + 1) * patch_size
                                lower = (max_row + 1) * patch_size

                                # 绘制红色矩形框
                                draw.rectangle([left, upper, right, lower], outline="red", width=2)

                            save_path = os.path.join(save_root, layer_name, channel_name)
                            os.makedirs(save_path, exist_ok=True)
                            new_image_name = str(label) + "_" + image_name
                            result_img.save(os.path.join(save_path, new_image_name))
                            processed_any_image = True
                            break

                
                pbar.update(1)

def process_all_layers(base_path, selected_images):
    save_root = f"/data/ty45972/taming-transformers/codebook_explanation_classification/results/Explanation/neuron/Net{args.model}/neuron_activation_images"
    for layer_folder in os.listdir(base_path):
        if not layer_folder.startswith('conv2_'):
            continue
        layer_path = os.path.join(base_path, layer_folder)
        if os.path.isdir(layer_path):
            process_and_save_images(selected_images, layer_path, save_root)
            print(layer_folder)

if __name__ == "__main__":
    base_path = f"/data/ty45972/taming-transformers/codebook_explanation_classification/results/Explanation/neuron/Net{args.model}/neuron_activation_results"
    for label in args.labels:
        print(f"start to process label {label}")
        # 读取第一个csv文件以选择指定标签的图片
        first_layer_path = os.path.join(base_path, os.listdir(base_path)[0])
        first_csv_path = os.path.join(first_layer_path, os.listdir(first_layer_path)[0])
        selected_images = load_and_select_images(first_csv_path, label=label, num_images=args.num_images)
        print(f"Selected {len(selected_images)} images for label {label}")
        # 遍历所有层并生成激活图像
        process_all_layers(base_path, selected_images)
