# generate_embeddings.py
import csv
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import logging
import sys
import io
import requests
from PIL import ImageDraw, ImageFont
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from omegaconf import OmegaConf
from IPython.display import display, display_markdown

# 添加父目录到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taming.models.new_vqgan import VQModel, GumbelVQ

# 配置日志
logging.basicConfig(filename='error_log.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# 解析命令行参数
parser = argparse.ArgumentParser(description="Generate Embeddings for concept model")
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--token_num', type=int, default=16384, help='Token number for processing (1024 or 16384)')
args = parser.parse_args()

# 禁用梯度以节省内存
torch.set_grad_enabled(False)
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 加载配置
def load_config(config_path):
    return OmegaConf.load(config_path)

# 加载VQGAN模型
def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()

# 预处理图像
def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x

# 图像编码器
def img_encoder(x, model):
    z, _, [_, _, indices], distance = model.encode(x)
    return z, indices, distance

# 根据token_num选择模型
if args.token_num == 16384:
    config = load_config("../logs/vqgan_imagenet_f16_16384/configs/model.yaml")
    model = load_vqgan(config, ckpt_path="../logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)
elif args.token_num == 1024:
    config = load_config("../logs/vqgan_imagenet_f16_1024/configs/model.yaml")
    model = load_vqgan(config, ckpt_path="../logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)
else:
    raise ValueError("Invalid token number. Choose either 1024 or 16384.")

# 设置输出路径
OUTPUT_PATH = "/data/ty45972/taming-transformers/codebook_explanation_concept/datasets/CUB/image_embedding/"

def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    if s < target_image_size:
        raise ValueError(f'Min dimension for image {s} < {target_image_size}')
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img

def process_image(image_path, size=256):
    image = Image.open(image_path)
    processed_img = preprocess(image, target_image_size=size, map_dalle=False).to(DEVICE)
    with torch.no_grad():
        z, indices, _ = img_encoder(processed_img, model)
    return z.squeeze(0).cpu().numpy(), indices.cpu().numpy()

def save_embeddings(embedding, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)

def process_and_save_embeddings(csv_file, output_dir, output_csv):
    output_csv_path = "/data/ty45972/taming-transformers/codebook_explanation_concept/datasets/CUB/image_embedding/" + output_csv
    df = pd.read_csv(csv_file)
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['embedding', 'label', 'indices'])
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path, label = row['image'], row['label']
            try:
                embedding, indices = process_image(os.path.join(DATASET_PATH, img_path))
                embedding_filename = os.path.basename(img_path).replace('.jpg', '.npy')
                output_path = os.path.join(output_dir, embedding_filename)
                save_embeddings(embedding, output_path)
                writer.writerow([os.path.join(embedding_filename), label, indices.tolist()])
            except Exception as e:
                logging.error(f"Skipping {img_path}: {e}")

if __name__ == "__main__":
    DATASET_PATH = ""
    
    TRAIN_CSV = f"/data/ty45972/taming-transformers/codebook_explanation_concept/datasets/CUB/concept_filtering_dataset/train_images.csv"
    VAL_CSV = f"/data/ty45972/taming-transformers/codebook_explanation_concept/datasets/CUB/concept_filtering_dataset/val_images.csv"
    TEST_CSV = f"/data/ty45972/taming-transformers/codebook_explanation_concept/datasets/CUB/concept_filtering_dataset/test_images.csv"



    process_and_save_embeddings(TRAIN_CSV, os.path.join(OUTPUT_PATH, "train"), "train_embeddings.csv")
    process_and_save_embeddings(VAL_CSV, os.path.join(OUTPUT_PATH, "validation"), "val_embeddings.csv")
    process_and_save_embeddings(TEST_CSV, os.path.join(OUTPUT_PATH, "test"), "test_embeddings.csv")
