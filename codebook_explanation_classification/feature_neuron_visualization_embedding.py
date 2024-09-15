import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import os
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
import random
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from taming.models.new_vqgan import VQModel, GumbelVQ
from main import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def load_model(config_path, checkpoint_path, device=None, eval_mode=True):
    # 加载配置文件
    config = OmegaConf.load(config_path)
    
    # 实例化模型
    model = instantiate_from_config(config.model)
    
    # 加载模型权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
            print(f"Loaded model from global step {global_step}.")
        else:
            global_step = None
        
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError("Checkpoint does not contain state_dict.")
    else:
        raise ValueError(f"Checkpoint file {checkpoint_path} not found.")
    
    # 指定device（默认为CUDA或CPU）
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将模型移动到指定的设备
    model = model.to(device)
    
    # 设置模型为评估模式（如果需要）
    if eval_mode:
        model.eval()
    
    return model

def chw_to_pillow(x):
    return Image.fromarray((255 * rescale(x.detach().cpu().numpy().transpose(1, 2, 0))).clip(0, 255).astype(np.uint8))

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

def l2_regularization(embedding):
    return torch.mean(embedding ** 2)

def high_frequency_penalty(decoded_img):
    img_fft = torch.fft.fftn(decoded_img, dim=(-2, -1))
    high_freq_mask = torch.ones_like(img_fft)
    high_freq_mask[..., :decoded_img.shape[-2] // 4, :decoded_img.shape[-1] // 4] = 0
    high_freq_mask[..., -decoded_img.shape[-2] // 4:, -decoded_img.shape[-1] // 4:] = 0
    high_freq_loss = torch.sum(torch.abs(img_fft * high_freq_mask))
    return high_freq_loss

def total_variation_penalty(image):
    diff1 = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    diff2 = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    return torch.sum(diff1) + torch.sum(diff2)

def compute_loss(layer_output, embedding, VQ_model, c=None, neuron_index=None, is_conv=True):
    # 计算激活损失（需要最大化，所以取负值）
    if c is not None:
        activation = -torch.max(layer_output[0, c, :, :])
    else:
        activation = -layer_output[0, neuron_index]

    # 计算其他损失
    l2_loss = l2_regularization(embedding)
    # hf_loss, tv_loss = 0, 0
    
    # if not is_conv:
    #     _, _, token_feature_map = decode_input_embedding(embedding, VQ_model)
    #     hf_loss = high_frequency_penalty(token_feature_map)
    #     print(f"hf_loss is {hf_loss}")
    #     tv_loss = total_variation_penalty(token_feature_map)

    # 计算每个损失的梯度
    # activation_loss_grad  = torch.autograd.grad(activation, embedding, retain_graph=True)[0]
    # l2_loss_grad = torch.autograd.grad(l2_loss, embedding, retain_graph=True)[0]
    # hf_loss_grad = torch.autograd.grad(hf_loss, embedding, retain_graph=True)[0] if hf_loss != 0 else torch.tensor(0.0, device=embedding.device)
    # tv_loss_grad = torch.autograd.grad(tv_loss, embedding, retain_graph=True)[0] if tv_loss != 0 else torch.tensor(0.0, device=embedding.device)

    # 计算每个梯度的模长
    # grad_norms = {
        # 'act': torch.norm(activation_loss_grad),
    #     'l2': torch.norm(l2_loss_grad) * 0.01,
        # 'hf': torch.norm(hf_loss_grad) if hf_loss != 0 else 0,
        # 'tv': torch.norm(tv_loss_grad) if tv_loss != 0 else 0
    # }
    # print(grad_norms)

    # # 动态调整权重：将梯度的模长倒数作为权重，平衡各损失的影响
    # total_grad_norm = sum(grad_norms.values())
    # adjusted_reg_factors = {key: (total_grad_norm / (value + 1e-8)) for key, value in grad_norms.items()}
    if is_conv:
        new_activation_loss = 0.01 * activation
    else:
        new_activation_loss = 0.1 * activation

    new_l2_loss = 0.1 * l2_loss
    # new_hf_loss = adjusted_reg_factors['hf'] * hf_loss
    # new_tv_loss = adjusted_reg_factors['tv'] * tv_loss

    # 计算总损失：最大化activation_loss，最小化其他损失
    total_loss = new_activation_loss + new_l2_loss

    return total_loss, new_activation_loss, new_l2_loss, activation

def generate_max_activation_embedding(model, device, VQ_model, input_shape=(1, 256, 16, 16), lr=0.01, num_steps=10000, num_neurons=128, threshold=1):
    def get_layer_outputs(layer, x):
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        handle = layer.register_forward_hook(hook)
        model(x)
        handle.remove()
        return outputs[0]

    for name, layer in model.named_modules():
        print(f"Optimizing layer: {name}")
        if isinstance(layer, nn.Conv2d):
            # 优化所有channel
            with torch.no_grad():
                sample_input = torch.randn(input_shape, device=device)
                layer_output = get_layer_outputs(layer, sample_input)

            num_channels = layer_output.shape[1]

            progress_bar = tqdm(total=num_channels, desc=f"Optimizing for layer {name}")

            for c in range(num_channels):
                input_embedding = torch.randn(input_shape, device=device, requires_grad=True)
                optimizer = torch.optim.Adam([input_embedding], lr=lr)

                for step in range(num_steps):
                    optimizer.zero_grad()
                    layer_output = get_layer_outputs(layer, input_embedding)

                    total_loss, activation_loss, l2_loss, activation = compute_loss(
                        layer_output, input_embedding, VQ_model, c=c, is_conv=True)

                    if step == 100 and abs(activation.item()) < threshold:
                        print(f"Skipping channel {c} in layer {name} due to low activation")
                        break

                    if step % 100 == 0:
                        print(f"Step {step}, total loss: {total_loss.item()}")
                        print(f"Activation: {activation.item()}")
                        print(f"L2 loss is {l2_loss}")

                    total_loss.backward()
                    optimizer.step()

                if abs(activation.item()) >= threshold:
                    _, _, token_feature_map = decode_input_embedding(input_embedding, VQ_model)
                    
                    layer_path = f"feature_visualization_map/{args.data_type}_data/neuron/{name}/"
                    os.makedirs(layer_path, exist_ok=True)
                    token_feature_map = chw_to_pillow(token_feature_map[0])
                    token_feature_map.save(layer_path + f"channel_{c}.png")

                progress_bar.update(1)

            progress_bar.close()

        elif isinstance(layer, nn.Linear):
            # 随机优化128个neuron
            with torch.no_grad():
                sample_input = torch.randn(input_shape, device=device)
                layer_output = get_layer_outputs(layer, sample_input)

            num_neurons = min(num_neurons, layer_output.shape[1])
            random_neurons = random.sample(range(layer_output.shape[1]), num_neurons)
            
            progress_bar = tqdm(total=num_neurons, desc=f"Optimizing for layer {name}")

            for neuron in random_neurons:
                input_embedding = torch.randn(input_shape, device=device, requires_grad=True)
                optimizer = torch.optim.Adam([input_embedding], lr=lr)

                for step in range(num_steps):
                    optimizer.zero_grad()
                    layer_output = get_layer_outputs(layer, input_embedding)

                    total_loss, activation_loss, l2_loss, activation = compute_loss(
                        layer_output, input_embedding, VQ_model, neuron_index=neuron, is_conv=False)

                    if step == 100 and abs(activation.item()) < threshold:
                        print(f"Skipping neuron {neuron} in layer {name} due to low activation")
                        break

                    if step % 100 == 0:
                        print(f"Step {step}, total loss: {total_loss.item()}")
                        print(f"Activation: {activation.item()}")
                        print(f"L2 loss is {l2_loss}")

                    total_loss.backward()
                    optimizer.step()

                if abs(activation.item()) >= threshold:
                    _, _, token_feature_map = decode_input_embedding(input_embedding, VQ_model)

                    layer_path = f"feature_visualization_map/{args.data_type}_data/neuron/{name}/"
                    os.makedirs(layer_path, exist_ok=True)
                    token_feature_map = chw_to_pillow(token_feature_map[0])
                    token_feature_map.save(layer_path + f"neuron_{neuron}.png")

                progress_bar.update(1)

            progress_bar.close()

def load_config(config_path):
    return OmegaConf.load(config_path)

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

    d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(codebook.weight**2, dim=1) - 2 * \
        torch.einsum('bd,dn->bn', z_flattened, rearrange(codebook.weight, 'n d -> d n'))

    min_encoding_indices = torch.argmin(d, dim=1)

    z_q = codebook(min_encoding_indices).view(z.shape)

    z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
    token_feature_map = VQ_model.decode(z_q)
    return z_q, min_encoding_indices, token_feature_map

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.data_type == "generated":
        classification_model_path = f"/data/ty45972/taming-transformers/codebook_explanation_classification/checkpoints/generated_data/ClassificationNet{args.model}/best_model.pth"
        classification_model = load_classification_model(args.model, classification_model_path, device)
    elif args.data_type == "original":
        classification_model_path = f"/data/ty45972/taming-transformers/codebook_explanation_classification/checkpoints/original_data/ClassificationNet{args.model}/best_model.pth"
        classification_model = load_classification_model(args.model, classification_model_path, device)  
    else:
        print("please select the correct data type")

    if args.data_type == "generated":
        config_path = "/data/ty45972/taming-transformers/logs/2021-04-03T19-39-50_cin_transformer/configs/2021-04-03T19-39-50-project.yaml"
        checkpoint_path = "/data/ty45972/taming-transformers/logs/2021-04-03T19-39-50_cin_transformer/checkpoints/last.ckpt"
        model = load_model(config_path, checkpoint_path, device=device, eval_mode=True)
        VQ_model = model.first_stage_model

    elif args.data_type == "original":
        config = load_config("../logs/vqgan_imagenet_f16_16384/configs/model.yaml")
        VQ_model = load_vqgan(config, ckpt_path="../logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(device)
    else:
        print("please select the correct data type")

    generate_max_activation_embedding(classification_model, device, VQ_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings to maximize neuron activations.')
    parser.add_argument('--gpu', type=int, default=1, help='Specify which GPU to use for computation')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], default=1,
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    parser.add_argument('--data_type', type=str, default="original", help='Specify which data type (original or generated)')
    args = parser.parse_args()
    
    main(args)
