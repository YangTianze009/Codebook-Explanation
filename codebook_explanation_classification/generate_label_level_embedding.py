'''
This file is used to generate the aggregated embedding for each label based on top n token activation results
'''

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from omegaconf import OmegaConf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from taming.models.new_vqgan import VQModel
import csv
import pickle


def load_config(config_path):
    return OmegaConf.load(config_path)

# 加载VQGAN模型
def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()

def load_top_tokens(csv_path, top_n, max_rows):
    """
    从指定的csv文件中加载Top N的前几行token及其对应的文件列表。

    参数:
    csv_path (str): 要读取的csv文件路径。
    top_n (int): 要查询的Top N级别（如1, 5, 10, 20）。
    max_rows (int): 要读取的最大行数。

    返回:
    token_list (list): (token索引, 权重)的列表。
    """
    token_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        in_top_n_section = False
        current_row = 0 # 从1开始计数
        for row in reader:
            # 检查是否到了Top N的部分
            if f"Top {top_n} Tokens" in row:
                in_top_n_section = True
                current_row = 0  # 重置行计数，从1开始
                next(reader)
                continue

            # 如果在Top N部分，开始读取前max_rows行
            if in_top_n_section and current_row < max_rows:
                token = int(row[0])  # 获取token索引
                frequency = int(row[1])  # 获取频率
                token_list.append((token, frequency))
                current_row += 1

            # 如果读取完指定行数，则退出
            if current_row > max_rows:
                break

    return token_list


def compute_weighted_embedding(token_list, codebook):
    """
    根据token列表及其频率加权生成聚合的embedding。

    参数:
    - token_list (list): (token索引, 权重)的列表
    - codebook (torch.Tensor): 形状为(16384, 256)的tensor，存储所有token的embedding

    返回:
    - weighted_embedding (torch.Tensor): 加权后的embedding，形状为(256,)
    """
    total_embedding = torch.zeros(codebook.size(1)).to(codebook.device)  # 初始化一个256维零向量
    total_weight = 0

    for token, weight in token_list:
        embedding = codebook[token]  # 从codebook中获取对应token的embedding
        total_embedding += embedding * weight  # 加权
        total_weight += weight
    if total_weight > 0:
        return total_embedding / total_weight  # 归一化
    else:
        return total_embedding  # 避免除以0

def generate_label_embeddings(folder_path, top_ns, max_rows_per_top_n, codebook):
    """
    生成每个label的embedding字典。

    参数:
    - folder_path (str): 存储label_n.csv文件的文件夹路径
    - top_ns (list): 需要读取的Top N级别列表（如[1, 5, 10, 20]）
    - max_rows_per_top_n (int): 从每个Top N部分读取的行数
    - codebook (torch.Tensor): 存储所有token embedding的tensor，形状为(16384, 256)

    返回:
    - label_embeddings (dict): 每个label的embedding字典
    """
    label_embeddings = {}

    for file_name in os.listdir(folder_path):
        if file_name.startswith('label_') and file_name.endswith('.csv'):
            label = int(file_name.split('_')[1].split('.')[0])  # 提取label编号
            csv_path = os.path.join(folder_path, file_name)

            # 初始化每个label的字典
            label_embeddings[label] = {}

            for top_n in top_ns:
                # 加载指定Top N部分的前max_rows行
                token_list = load_top_tokens(csv_path, top_n, max_rows_per_top_n)
                # 计算加权聚合embedding
                label_embeddings[label][f'top{top_n}'] = compute_weighted_embedding(token_list, codebook)

    return label_embeddings

def save_embeddings_as_pkl(label_embeddings, save_path):
    """
    将label embeddings保存为pkl文件。

    参数:
    - label_embeddings (dict): 要保存的label embeddings字典
    - save_path (str): pkl文件的保存路径
    """
    with open(save_path, 'wb') as f:
        pickle.dump(label_embeddings, f)
    print(f"Label embeddings saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings to maximize label activation.')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    config = load_config("../logs/vqgan_imagenet_f16_16384/configs/model.yaml")
    VQ_model = load_vqgan(config, ckpt_path="../logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(device)
    
    # 获取codebook
    codebook = VQ_model.quantize.embedding.weight.to(device)

    # 假设codebook已经定义好，是一个torch.Tensor，例如：codebook = torch.randn(16384, 256)
    # 调用函数
    folder_path = "/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/generated_data/label/Net1/label_activation_statistics"
    top_ns = [1, 5, 10, 20]  # 需要读取的Top N级别
    max_rows_per_top_n = 50  # 读取每个Top N部分的前35行

    label_embeddings = generate_label_embeddings(folder_path, top_ns, max_rows_per_top_n, codebook)
    save_path_base = "/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/generated_data/similarity_results/"
    os.makedirs(save_path_base, exist_ok=True)
    save_embeddings_as_pkl(label_embeddings, save_path_base + "label_embeddings.pkl")
    # print(label_embeddings)
