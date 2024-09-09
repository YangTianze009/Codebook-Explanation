import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_file, data_folder):
        self.data_folder = data_folder
        self.data = []
        self.labels = []

        # 读取 CSV 文件，提取文件名和标签
        with open(embeddings_file, 'r') as f:
            lines = f.readlines()
            # 跳过标题行
            for line in lines[1:]:
                parts = line.strip().split(',')
                filename = parts[0]  # 读取文件名
                label = int(parts[1])  # 读取标签
                self.data.append(filename)  # 将文件名添加到数据列表
                self.labels.append(label)  # 将标签添加到标签列表

    def __len__(self):
        return len(self.data)  # 返回数据集的大小

    def __getitem__(self, idx):
        file_path = f"{self.data_folder}/{self.data[idx]}"  # 构建文件路径
        embedding = np.load(file_path)  # 加载 .npy 文件中的嵌入向量
        label = self.labels[idx]  # 获取对应的标签
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  # 返回 PyTorch 张量形式的嵌入向量和标签

def get_train_val_dataloaders(train_csv, val_csv, data_folder, batch_size, shuffle=True):
    train_dataset = EmbeddingDataset(f"{data_folder}/" + train_csv, f"{data_folder}/train")  # 创建训练集数据集实例
    val_dataset = EmbeddingDataset(f"{data_folder}/" + val_csv, f"{data_folder}/val")  # 创建验证集数据集实例

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)  # 训练集数据加载器，支持打乱数据
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集数据加载器，不打乱数据

    return train_loader, val_loader  # 返回训练集和验证集数据加载器

def get_test_dataloader(test_csv, data_folder, batch_size):
    test_dataset = EmbeddingDataset(f"{data_folder}/" + test_csv, f"{data_folder}/test")  # 创建测试集数据集实例

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集数据加载器，不打乱数据

    return test_loader  # 返回测试集数据加载器
