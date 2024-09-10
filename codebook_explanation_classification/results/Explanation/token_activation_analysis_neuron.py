import csv
import os
import numpy as np
from tqdm import tqdm
import sys
import argparse
import ast
from collections import defaultdict
import torch

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
args = parser.parse_args()

def load_token_indices(embedding_csv_path):
    token_indices_dict = {}
    with open(embedding_csv_path, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        for row in reader:
            npy_file = row[0]
            token_indices = ast.literal_eval(row[2])
            token_indices_dict[npy_file] = token_indices
    return token_indices_dict

def is_channel_activated(csv_data, threshold=1e-3):
    """快速检测是否有任何激活值超过阈值"""
    for row in csv_data:
        activation_values = np.array(ast.literal_eval(row[2]))
        if np.any(activation_values > threshold):  # 检查是否有任何激活值超过阈值
            return True
    return False

def process_and_save_channel_statistics(layer_folder, save_root, token_indices_dict):
    layer_name = os.path.basename(layer_folder)
    csv_files = [f for f in os.listdir(layer_folder) if f.endswith('.csv')]

    os.makedirs(save_root, exist_ok=True)

    with tqdm(total=len(csv_files), desc=f"Processing {layer_name}", unit="csv file") as pbar:
        for csv_file in csv_files:
            csv_path = os.path.join(layer_folder, csv_file)
            channel_name = os.path.splitext(csv_file)[0]

            # 预先加载CSV文件中的所有数据
            with open(csv_path, 'r') as infile:
                reader = csv.reader(infile)
                csv_data = list(reader)[1:]  # 跳过header

            # 快速检测该channel是否被激活
            if not is_channel_activated(csv_data):
                print(f"Skipping inactive channel: {channel_name}")
                continue

            # 如果channel被激活，则继续进行详细处理
            label_statistics = defaultdict(lambda: {"total_images": 0, "tokens": defaultdict(int), "combinations": defaultdict(int)})
            token_frequency = defaultdict(int)
            combination_frequency = defaultdict(lambda: {"count": 0, "images": []})

            for row in csv_data:
                npy_file = row[0]
                label = int(row[1])
                activation_values = np.array(ast.literal_eval(row[2]))
                max_activation = activation_values.max()
                max_activation_indices = np.where(activation_values == max_activation)[0]

                token_indices = token_indices_dict.get(npy_file)
                if token_indices is None:
                    continue

                max_tokens = [token_indices[i] for i in max_activation_indices]
                
                # 更新label的图片计数
                label_statistics[label]["total_images"] += 1

                # 统计单个token
                for token in max_tokens:
                    label_statistics[label]["tokens"][token] += 1
                    token_frequency[token] += 1

                # 统计组合，顺序不能打乱
                combination = tuple(max_tokens)
                label_statistics[label]["combinations"][combination] += 1
                combination_frequency[combination]["count"] += 1
                combination_frequency[combination]["images"].append(npy_file)

            # 保存每个label的统计信息
            save_path = os.path.join(save_root, f"{channel_name}_label.csv")
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Label", "Total Images", "Top Tokens (Token, Frequency)", "Top Combinations (Combination, Frequency)"])
                
                for label, stats in label_statistics.items():
                    total_images = stats["total_images"]
                    if total_images == 0:
                        continue

                    # 只保存前5个最大激活token
                    top_tokens = sorted(stats["tokens"].items(), key=lambda x: x[1], reverse=True)[:5]
                    token_str = "; ".join([f"{token}:{freq}" for token, freq in top_tokens])

                    # 只保存前5个出现次数大于1的组合
                    top_combinations = [(comb, freq) for comb, freq in stats["combinations"].items() if freq > 1]
                    top_combinations = sorted(top_combinations, key=lambda x: x[1], reverse=True)[:5]
                    if top_combinations:
                        combination_str = "; ".join([f"{comb}:{freq}" for comb, freq in top_combinations])
                    else:
                        combination_str = "None"

                    writer.writerow([label, total_images, token_str, combination_str])

            # 另存全局信息
            save_global_path = os.path.join(save_root, f"{channel_name}_all.csv")
            with open(save_global_path, 'w', newline='') as globalfile:
                writer = csv.writer(globalfile)
                writer.writerow(["Token", "Frequency"])
                top_global_tokens = sorted(token_frequency.items(), key=lambda x: x[1], reverse=True)[:max(1, int(len(token_frequency) * 0.05))]
                for token, freq in top_global_tokens:
                    writer.writerow([token, freq])

                writer.writerow([])
                writer.writerow(["Combination", "Frequency", "Image Files"])
                top_global_combinations = sorted(combination_frequency.items(), key=lambda x: x[1]["count"], reverse=True)[:max(1, int(len(combination_frequency) * 0.05))]
                for comb, data in top_global_combinations:
                    images_str = "; ".join(data["images"])
                    writer.writerow([comb, data["count"], images_str])

            pbar.update(1)

def process_all_layers(base_path, token_indices_dict):
    save_root = f"/data/ty45972/taming-transformers/codebook_explanation_classification/results/Explanation/neuron/Net{args.model}/neuron_activation_statistics"
    
    for layer_folder in os.listdir(base_path):
        if not layer_folder.startswith('conv2_') and not layer_folder.startswith('conv3_'):
            continue
        layer_path = os.path.join(base_path, layer_folder)
        if os.path.isdir(layer_path):
            process_and_save_channel_statistics(layer_path, os.path.join(save_root, layer_folder), token_indices_dict)
            print(f"Processed: {layer_folder}")

if __name__ == "__main__":
    base_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/neuron/Net{args.model}/neuron_activation_results"
    embedding_csv_path = "/data2/ty45972_data_2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_embeddings.csv"
    
    # 加载token indices
    token_indices_dict = load_token_indices(embedding_csv_path)

    # 处理所有层并生成统计信息
    process_all_layers(base_path, token_indices_dict)
