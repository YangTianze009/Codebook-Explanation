import os
import io
import requests
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from omegaconf import OmegaConf
from taming.models.new_vqgan import VQModel, GumbelVQ
from dall_e import map_pixels, unmap_pixels, load_model
from IPython.display import display, display_markdown
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Patch Generation Pipeline")
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--size', type=int, default=256, help='Image size for processing')
parser.add_argument('--token_num', type=int, default=1024, help='token number for processing')
parser.add_argument('--dataset', type=str, default="coco", help='dataset for processing')
args = parser.parse_args()

# Disable grad to save memory
torch.set_grad_enabled(False)
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"device is {DEVICE}")

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if x.mode != "RGB":
        x = x.convert("RGB")
    return x

def img_encoder(x, model):
    z, _, [_, _, indices], distance = model.encode(x)
    # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    return indices, distance

if args.token_num == 16384:
    config16384 = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
    model = load_vqgan(config16384, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)

elif args.token_num == 1024:
    config1024 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    model = load_vqgan(config1024, ckpt_path="logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)

else:
    print("Token number doesn't exist")


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    if s < target_image_size:
        raise ValueError(f'Min dimension for image {s} < {target_image_size}')
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle:
        img = map_pixels(img)
    return img

def patch_generation_pipeline(img_path, size=256):
    img = Image.open(img_path)
    processed_img = preprocess(img, target_image_size=size, map_dalle=False)
    # print(f"Processed image shape for {img_path}: {processed_img.shape}")
    x_vqgan = processed_img.to(DEVICE)
    indices, distance = img_encoder(preprocess_vqgan(x_vqgan), model)
    original_img = custom_to_pil(preprocess_vqgan(x_vqgan[0]))
    return indices, distance, original_img

def save_patches(indices, distance, original_img, image_name, size):
    for i, index in enumerate(indices.flatten()):
        row, col = divmod(i, 16)
        patch = original_img.crop((col * 16, row * 16, (col + 1) * 16, (row + 1) * 16))
        if args.dataset == "coco":
            patch_dir = os.path.join(f"patches/coco/{args.token_num}_embedding_patches/img_size_{size}", str(index.item()))
        if args.dataset == "imagenet":
            patch_dir = os.path.join(f"patches/imagenet_validation/{args.token_num}_embedding_patches/img_size_{size}", str(index.item()))
        os.makedirs(patch_dir, exist_ok=True)
        # patch_path = os.path.join(patch_dir, f"{category}_{image_name}_{distance.flatten()[i].item():.4f}.png")
        patch_path = os.path.join(patch_dir, f"{image_name}_{distance.flatten()[i].item():.4f}.png")
        patch.save(patch_path)

import logging
# 配置日志
logging.basicConfig(filename='error_log.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

'''
generate patches for different categories
'''
# def process_images_in_folder(folder_path):
#     unique_indices = set()
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#     for category in tqdm(subfolders, desc="Processing categories"):
#         category_path = os.path.join(folder_path, category)
#         images = [img for img in os.listdir(category_path) if img.endswith('.jpg')]
#         for image_name in images:
#             img_path = os.path.join(category_path, image_name)
#             try:
#                 indices, distance, original_img = patch_generation_pipeline(img_path)
#                 unique_indices.update(indices.flatten().tolist())
#                 save_patches(indices, distance, original_img, category, os.path.splitext(image_name)[0])
#             except Exception as e:
#                 logging.error(f"Skipping {img_path}: {e}")

#     print("Processing completed. Check error_log.log for skipped images.")
#     print(f"Total unique indices: {len(unique_indices)}")


'''
Generate patches for the whole dataset
'''
def process_images_in_folder(folder_path, size=256):
    print(f"current image size is {size}")
    unique_indices = set()
    if args.dataset == "coco":
        images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
    elif args.dataset == "imagenet":
        images = [img for img in os.listdir(folder_path) if img.endswith('.JPEG')]
    for image_name in tqdm(images):
        img_path = os.path.join(folder_path, image_name)
        try:
            indices, distance, original_img = patch_generation_pipeline(img_path, size)
            unique_indices.update(indices.flatten().tolist())
            save_patches(indices, distance, original_img, os.path.splitext(image_name)[0], size)
        except Exception as e:
            logging.error(f"Skipping {img_path}: {e}")

    print("Processing completed. Check error_log.log for skipped images.")
    print(f"Total unique indices: {len(unique_indices)}")

if __name__ == "__main__":
    if args.dataset == "coco":
        process_images_in_folder("/data/ty45972/taming-transformers/datasets/coco/train2017", size=args.size)
    elif args.dataset == "imagenet":
        process_images_in_folder("/data/ty45972/taming-transformers/datasets/ILSVRC2012_img_val", size=args.size)
    else:
        print("Dataset not exist")