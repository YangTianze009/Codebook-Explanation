# split_dataset.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 配置路径
DATASET_PATH = "/data/ty45972/taming-transformers/datasets/imagenet"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
MAPPING_CSV = "datasets/class_mapping.csv"

def create_class_mapping(train_path):
    classes = sorted(os.listdir(train_path))  # 确保标签按字典顺序排列
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    return class_to_idx

def save_class_mapping(class_to_idx, csv_path):
    df = pd.DataFrame(list(class_to_idx.items()), columns=['class', 'label'])
    df.to_csv(csv_path, index=False)

def split_train_val_per_class(train_path, class_to_idx, split_ratio=0.1):
    train_data = []
    val_data = []

    for cls in class_to_idx.keys():
        cls_path = os.path.join(train_path, cls)
        images = [img for img in os.listdir(cls_path) if img.endswith('.JPEG')]
        images = [os.path.join(cls, img) for img in images]

        train_images, val_images = train_test_split(images, test_size=split_ratio, random_state=42)
        train_data.extend([('train/' + img, class_to_idx[cls]) for img in train_images])
        val_data.extend([('train/' + img, class_to_idx[cls]) for img in val_images])

    return train_data, val_data

def get_test_data(val_path, class_to_idx):
    data = []
    for cls in class_to_idx.keys():
        cls_path = os.path.join(val_path, cls)
        images = [img for img in os.listdir(cls_path) if img.endswith('.JPEG')]
        for img in images:
            data.append((os.path.join('val', cls, img), class_to_idx[cls]))
    return data

def save_to_csv(data, csv_path):
    df = pd.DataFrame(data, columns=['image', 'label'])
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    class_to_idx = create_class_mapping(TRAIN_PATH)
    save_class_mapping(class_to_idx, MAPPING_CSV)

    train_data, val_data = split_train_val_per_class(TRAIN_PATH, class_to_idx, split_ratio=0.1)
    test_data = get_test_data(VAL_PATH, class_to_idx)
    
    save_to_csv(train_data, "datasets/training_images.csv")
    save_to_csv(val_data, "datasets/validation_images.csv")
    save_to_csv(test_data, "datasets/testing_images.csv")
