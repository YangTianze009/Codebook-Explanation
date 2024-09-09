import os
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

def extract_number_from_filename(filename):
    # Extract the number from the filename, e.g., airplane_72_14.5166.png or airplane_72_14.png
    match = re.search(r'_(\d+(\.\d+)?)\.png$', filename)
    if match:
        return float(match.group(1))
    else:
        return None

def find_top_patches(folder_path, patch_number):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            number = extract_number_from_filename(filename)
            if number is not None:
                images.append((number, filename))
    
    # Sort images by the extracted number and select the top 256
    images_sorted = sorted(images, key=lambda x: x[0])
    return images_sorted[:(patch_number*patch_number)]

def merge_images(image_paths, output_path, csv_path, patch_number):
    patch_size = 16
    merged_image = np.zeros((patch_size * patch_number, patch_size * patch_number, 3), dtype=np.uint8)
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Row', 'Col', 'Filename'])

        for idx, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            row = idx // patch_number
            col = idx % patch_number
            merged_image[row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size] = img
            writer.writerow([row, col, image_path.name])

    cv2.imwrite(output_path, merged_image)

def process_folders(base_path, output_base_path, patch_number):
    base_path = Path(base_path)
    output_base_path = Path(output_base_path)

    subfolders = [subfolder for subfolder in base_path.iterdir() if subfolder.is_dir()]
    
    for subfolder in tqdm(subfolders, desc="Processing folders"):
        top_images = find_top_patches(subfolder, patch_number)
        top_image_paths = [subfolder / img[1] for img in top_images]

        # Create output directory if it doesn't exist
        output_folder = output_base_path / subfolder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Copy top 256 images to the output folder
        # for img_path in top_image_paths:
        #     output_img_path = output_folder / img_path.name
        #     cv2.imwrite(str(output_img_path), cv2.imread(str(img_path)))

        # Merge top patch_number * patch_number patches into a image
        merged_image_path = output_folder / 'all_merged_image.png'
        csv_path = output_folder / 'all_patches_info.csv'
        merge_images(top_image_paths, str(merged_image_path), str(csv_path), patch_number)

if __name__ == "__main__":
    img_size = 256
    downsampling_rate = 16
    patch_number = int(img_size / downsampling_rate)
    
    dataset_name = "coco"
    token_number = 1024
    base_path = f"/data/ty45972/taming-transformers/patches/{dataset_name}/{token_number}_embedding_patches/img_size_{img_size}"
    output_base_path = f"/data/ty45972/taming-transformers/patches/{dataset_name}/{token_number}_merged_patches/img_size_{img_size}"

    process_folders(base_path, output_base_path, patch_number)
