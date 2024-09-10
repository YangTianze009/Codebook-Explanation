import csv
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse
import ast

parser = argparse.ArgumentParser(description="Visual Token contributions")
parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
parser.add_argument('--model', type=int, choices=[1, 2, 3], required=True,
                    help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
parser.add_argument('--data', type=str, choices=["generated", "original"], default="generated",
                    help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')

args = parser.parse_args()

def load_token_indices(embedding_csv_path):
    # 生成保存token indices的文件名
    token_indices_save_path = embedding_csv_path.replace("train_embeddings.csv", "train_token_indices.pkl")
    
    # 检查是否已经存在保存的token_indices文件
    if os.path.exists(token_indices_save_path):
        print(f"Loading token indices from {token_indices_save_path}")
        with open(token_indices_save_path, 'rb') as f:
            token_indices_dict = pickle.load(f)
        return token_indices_dict
    
    # 如果没有保存的token_indices文件，就进行处理
    print(f"Processing token indices from {embedding_csv_path}")
    token_indices_dict = {}

    # 先计算文件中的总行数，以便显示进度条
    with open(embedding_csv_path, 'r') as infile:
        total_lines = sum(1 for _ in infile) - 1  # 减去header行

    # 重新打开文件并读取内容，同时显示进度条
    with open(embedding_csv_path, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # 跳过header
        for row in tqdm(reader, total=total_lines, desc="Loading token indices"):
            npy_file = row[0]
            token_indices = ast.literal_eval(row[2])
            token_indices_dict[npy_file] = token_indices
    
    # 保存处理后的token_indices_dict
    with open(token_indices_save_path, 'wb') as f:
        pickle.dump(token_indices_dict, f)
    print(f"Token indices saved to {token_indices_save_path}")
    
    return token_indices_dict


def process_csv_files(folder_path, save_root, token_indices_dict, top_n_list=[1, 5, 10, 20]):
    os.makedirs(save_root, exist_ok=True)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    file_count = len(csv_files)
    print(f"file count is {file_count}")
    
    # 初始化全局统计信息
    overall_token_statistics = {n: defaultdict(lambda: {"count": 0, "files": []}) for n in top_n_list}

    with tqdm(total=file_count, desc="Processing CSV files", unit="file") as pbar:
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            label_name = os.path.splitext(csv_file)[0]

            # 读取CSV文件数据
            with open(csv_path, 'r') as infile:
                reader = csv.reader(infile)
                csv_data = list(reader)[1:]  # 跳过header

            # 初始化统计信息
            label_token_statistics = {n: defaultdict(lambda: {"count": 0, "files": []}) for n in top_n_list}

            # 统计该label下的总图片数量
            total_images = len(csv_data)

            # 遍历每一行，统计token的最大激活值
            for row in csv_data:
                npy_file = row[0]
                contribution_values = np.array(ast.literal_eval(row[2]))
                
                # 直接从token_indices_dict中查找token索引
                token_indices = token_indices_dict.get(npy_file)
                if token_indices is None:
                    continue

                # 分别统计top 1, 5, 10, 20的最大激活token
                for n in top_n_list:
                    top_n_indices = np.argsort(contribution_values)[-n:][::-1]  # 获取前n个最大激活值的索引
                    top_tokens = [token_indices[i] for i in top_n_indices]
                    
                    # 统计出现频率并记录每个token出现在哪些文件中
                    for token in top_tokens:
                        label_token_statistics[n][token]["count"] += 1
                        label_token_statistics[n][token]["files"].append(npy_file)
                        overall_token_statistics[n][token]["count"] += 1
                        overall_token_statistics[n][token]["files"].append(npy_file)

            # 保存每个label的统计结果
            save_label_path = os.path.join(save_root, f"{label_name}.csv")
            with open(save_label_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # 先写入该label的总图片数
                writer.writerow([f"Total Images: {total_images}"])
                
                # 写入每个Top N的统计
                for n in top_n_list:
                    writer.writerow([f"Top {n} Tokens"])
                    writer.writerow(["Token", "Frequency", "Files"])

                    top_tokens = sorted(label_token_statistics[n].items(), key=lambda x: x[1]["count"], reverse=True)[:50]
                    for token, data in top_tokens:
                        files_str = "; ".join(data["files"])  # 将文件名合并成字符串
                        writer.writerow([token, data["count"], files_str])

            pbar.update(1)

    # 保存全局统计信息
    global_save_path = os.path.join(save_root, "global_top_tokens_statistics.csv")
    with open(global_save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Top N", "Token", "Frequency", "Files"])

        for n in top_n_list:
            writer.writerow([f"Top {n} Tokens"])
            top_tokens = sorted(overall_token_statistics[n].items(), key=lambda x: x[1]["count"], reverse=True)[:50]
            for token, data in top_tokens:
                files_str = "; ".join(data["files"])
                writer.writerow([token, data["count"], files_str])

    # 记录处理的文件总数
    print(f"Total files processed: {file_count}")


if __name__ == "__main__":
    if args.data == "generated":
        folder_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/generated_data/label/Net{args.model}/label_activation_results"  # 替换为你存放label_n.csv文件的文件夹路径
        save_root = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/generated_data/label/Net{args.model}/label_activation_statistics"  # 替换为保存结果的路径
        embedding_csv_path = "/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/train_embeddings.csv"  # 替换为training_embedding.csv的路径

    elif args.data == "original":
            folder_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/original_data/label/Net{args.model}/label_activation_results"  # 替换为你存放label_n.csv文件的文件夹路径
            save_root = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/original_data/label/Net{args.model}/label_activation_statistics"  # 替换为保存结果的路径
            embedding_csv_path = "/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_original/train_embeddings.csv"  # 替换为training_embedding.csv的路径

    # 加载或生成token indices
    token_indices_dict = load_token_indices(embedding_csv_path)

    # 处理CSV文件并生成统计信息
    process_csv_files(folder_path, save_root, token_indices_dict)
