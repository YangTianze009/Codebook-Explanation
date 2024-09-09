import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3
import os
from omegaconf import OmegaConf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from taming.models.new_vqgan import VQModel, GumbelVQ
from einops import rearrange
from PIL import Image

rescale = lambda x: (x + 1.) / 2.

def chw_to_pillow(x):
    return Image.fromarray((255*rescale(x.detach().cpu().numpy().transpose(1,2,0))).clip(0,255).astype(np.uint8))

def load_classification_model(model_choice, model_path, device):
    num_classes = 1000  # 假设有1000个类别，可以根据实际情况调整
    if model_choice == 1:
        model = ClassificationNet1(num_classes)
    elif model_choice == 2:
        model = ClassificationNet2(num_classes)
    elif model_choice == 3:
        model = ClassificationNet3(num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape, device=device)
    return -torch.log(-(torch.log(U + eps)) + eps)

def gumbel_softmax(logits, temperature, device='cpu', hard=True):
    gumbel_noise = sample_gumbel(logits.size(), device=device)
    y = logits + gumbel_noise
    y = torch.softmax(y / temperature, dim=-1)
    
    if hard:
        shape = y.size()
        _, max_idx = y.max(dim=-1, keepdim=True)  # 获取每行最大值对应的索引
        y_hard = torch.zeros_like(y).scatter_(-1, max_idx, 1.0)  # 将最大索引位置置为1，其余位置为0
        y = (y_hard - y).detach() + y  # 通过Straight-Through技巧保留梯度
    return y, max_idx.squeeze(-1)  # 返回选择的one-hot矩阵和索引

def generate_max_activation_image(model, target_label, device, codebook, num_steps=10000, lr=0.01, reg=0, temperature=1.0):
    # 初始化P矩阵, P的形状为 (256, 16384)，每行表示从codebook中选择token的概率分布
    P = torch.randn(256, 16384, device=device, requires_grad=True)
    
    # 定义优化器
    optimizer = torch.optim.Adam([P], lr=lr)
    
    final_token_indices = None  # 用于存储最终的token索引

    for step in tqdm(range(num_steps), desc="Optimizing tokens"):
        optimizer.zero_grad()
        
        # Gumbel-Softmax采样，硬选择出每行的token
        selected_tokens, token_indices = gumbel_softmax(P, temperature, device=device, hard=True)  # 硬选择，每一行选择一个token
        # print(f"token_indices is {token_indices}")
        
        # 从codebook中选择具体的token
        selected_embedding = torch.matmul(selected_tokens, codebook.to(device))  # (256, 256)
        selected_embedding = rearrange(selected_embedding, '(h w) d -> 1 d h w', h=16, w=16).to(device)  # reshape成ClassificationNet要求的形状
        
        # 前向传播
        output = model(selected_embedding)
        
        # 定义损失函数：最大化目标类别的激活值，并添加正则化项
        loss = -output[0, target_label] + reg * torch.mean(selected_embedding ** 2)  # 负的激活值，因为我们要最大化
    
        
        # 反向传播
        loss.backward()
        
        # 更新P矩阵
        optimizer.step()

        # 保存最终选择的token索引
        final_token_indices = token_indices.detach().cpu().numpy()

        # 打印损失
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            print(f"Current output is {output[0].shape}")
    
    # 返回最后一次优化得到的token索引
    return final_token_indices, selected_embedding

def load_config(config_path):
    return OmegaConf.load(config_path)

# 加载VQGAN模型
def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()

def decode_input_embedding(input_embedding, VQ_model):
    codebook = VQ_model.quantize.embedding

    z = rearrange(input_embedding, 'b c h w -> b h w c').contiguous()
    z_flattened = z.view(-1, 256)
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

    d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(codebook.weight**2, dim=1) - 2 * \
        torch.einsum('bd,dn->bn', z_flattened, rearrange(codebook.weight, 'n d -> d n'))

    min_encoding_indices = torch.argmin(d, dim=1)
    # print(f"indices are {min_encoding_indices}")

    z_q = codebook(min_encoding_indices).view(z.shape)

    # reshape back to match original input shape
    z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
    token_feature_map = VQ_model.decode(z_q)

    return z_q, min_encoding_indices, token_feature_map
def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    classification_model_path = f"/data/ty45972/taming-transformers/codebook_explanation_classification/checkpoints/generated_data/ClassificationNet{args.model}/best_model.pth"
    classification_model = load_classification_model(args.model, classification_model_path, device)
    
    config = load_config("../logs/vqgan_imagenet_f16_16384/configs/model.yaml")
    VQ_model = load_vqgan(config, ckpt_path="../logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(device)
    
    # 获取codebook
    codebook = VQ_model.quantize.embedding.weight.to(device)
    
    # 生成最大化目标类别激活值的图像
    target_label = args.label_index
    token_indices, optimized_embedding = generate_max_activation_image(classification_model, target_label, device, codebook)
    print(f"optimized token_indices is {token_indices}")
    token_embedding, new_token_indices, feature_maps = decode_input_embedding(optimized_embedding, VQ_model)
    feature_visualization_map_path = f"feature_visualization_map/{target_label}/"
    os.makedirs(feature_visualization_map_path, exist_ok=True)
    print(f"final indices is {new_token_indices}")
    for token_feature_map in feature_maps:
        token_feature_map = chw_to_pillow(token_feature_map)
        token_feature_map.save(feature_visualization_map_path + f"label_{args.label_index}_feature_map.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings to maximize label activation.')
    parser.add_argument('--label_index', type=int, required=True, help='Label index to maximize activation')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], required=True,
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    args = parser.parse_args()
    
    main(args)
