import torch
import torch.nn as nn
import numpy as np
import os
import sys
import csv
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torchvision
import argparse

# 添加必要的路径
current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3

parent_directory = os.path.abspath(os.path.join(current_directory, '../..'))
sys.path.append(parent_directory)

from omegaconf import OmegaConf
from main import instantiate_from_config
from einops import rearrange
from taming.models.new_vqgan import VQModel

def rescale(x):
    return (x + 1.) / 2.

parser = argparse.ArgumentParser(description='Replace tokens with baseline tokens for evaluation.')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
parser.add_argument('--model', type=int, choices=[1, 2, 3], required=True, help='Classification model to use')
args = parser.parse_args()

def chw_to_pillow(x):
    return Image.fromarray((255*rescale(x.detach().cpu().numpy().transpose(1,2,0))).clip(0,255).astype(np.uint8))

def load_classification_model(model_choice, model_path, device):
    num_classes = 1000
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

def load_pretrained_models(device):
    models = {
        'vit_b_16': torchvision.models.vit_b_16(pretrained=True),
        'vit_b_32': torchvision.models.vit_b_32(pretrained=True),
        'resnet18': torchvision.models.resnet18(pretrained=True),
        'resnet50': torchvision.models.resnet50(pretrained=True)
    }
    for model in models.values():
        model.to(device)
        model.eval()
    return models

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

def load_model(config_path, checkpoint_path, device=None, eval_mode=True):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError("Checkpoint does not contain state_dict.")
    else:
        raise ValueError(f"Checkpoint file {checkpoint_path} not found.")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    if eval_mode:
        model.eval()
    
    return model

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

def load_baseline_tokens(csv_path):
    tokens = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tokens.append(int(row['Token']))
    return tokens

def replace_tokens_with_baseline(initial_tokens, baseline_tokens, top_left, bottom_right):
    new_tokens = initial_tokens.copy()
    token_index = 0
    for i in range(top_left[0], bottom_right[0]+1):
        for j in range(top_left[1], bottom_right[1]+1):
            index = i * 16 + j
            if token_index < len(baseline_tokens):
                new_tokens[index] = baseline_tokens[token_index]
                token_index += 1
    return new_tokens

def process_label(initial_label, target_label, files, token_dict, classification_model, pretrained_models, VQ_model, device, baseline_tokens, top_left, bottom_right):
    results = {model_name: {'initial_original_logit': [], 'initial_target_logit': [], 
                            'initial_original_prob': [], 'initial_target_prob': [],
                            'final_original_logit': [], 'final_target_logit': [], 
                            'final_original_prob': [], 'final_target_prob': []}
               for model_name in pretrained_models}
    
    for file_name in tqdm(files, desc=f"Processing files for label {initial_label}", leave=False):
        original_image_file = f"/data/ty45972/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test/{file_name}"
        image_array = np.load(original_image_file)
        original_image_tensor = torch.tensor(image_array).unsqueeze(0).to(device)
        
        _, _, original_images = decode_input_embedding(original_image_tensor, VQ_model)
        original_image = chw_to_pillow(original_images[0])
        
        for model_name, pretrained_model in pretrained_models.items():
            with torch.no_grad():
                preprocessed_img = preprocess_image(original_image).to(device)
                output = pretrained_model(preprocessed_img)
                softmax_output = torch.softmax(output, dim=1)
                results[model_name]['initial_original_logit'].append(output[0, initial_label].item())
                results[model_name]['initial_target_logit'].append(output[0, target_label].item())
                results[model_name]['initial_original_prob'].append(softmax_output[0, initial_label].item())
                results[model_name]['initial_target_prob'].append(softmax_output[0, target_label].item())
        
        # 替换tokens
        initial_tokens = token_dict[file_name]
        new_tokens = replace_tokens_with_baseline(initial_tokens, baseline_tokens, top_left, bottom_right)
        new_embedding = VQ_model.quantize.embedding(torch.tensor(new_tokens).to(device)).view(1, 256, 16, 16)
        print("Changed tokens:")
        
        for i in range(top_left[0], bottom_right[0]+1):
            for j in range(top_left[1], bottom_right[1]+1):
                index = i * 16 + j
                if initial_tokens[index] != new_tokens[index]:
                    print(f"Position ({i}, {j}): {initial_tokens[index]} -> {new_tokens[index]}")
        print("------------------------")
        # 解码新的embedding
        _, _, new_images = decode_input_embedding(new_embedding, VQ_model)
        new_image = chw_to_pillow(new_images[0])
        
        for model_name, pretrained_model in pretrained_models.items():
            with torch.no_grad():
                preprocessed_img = preprocess_image(new_image).to(device)
                output = pretrained_model(preprocessed_img)
                softmax_output = torch.softmax(output, dim=1)
                results[model_name]['final_original_logit'].append(output[0, initial_label].item())
                results[model_name]['final_target_logit'].append(output[0, target_label].item())
                results[model_name]['final_original_prob'].append(softmax_output[0, initial_label].item())
                results[model_name]['final_target_prob'].append(softmax_output[0, target_label].item())

    return results

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    classification_model_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/checkpoints/generated_data/ClassificationNet{args.model}/best_model.pth"
    classification_model = load_classification_model(args.model, classification_model_path, device)
    
    pretrained_models = load_pretrained_models(device)

    config_path = "/data/ty45972/taming-transformers/logs/2021-04-03T19-39-50_cin_transformer/configs/2021-04-03T19-39-50-project.yaml"
    checkpoint_path = "/data/ty45972/taming-transformers/logs/2021-04-03T19-39-50_cin_transformer/checkpoints/last.ckpt"
    model = load_model(config_path, checkpoint_path, device=device, eval_mode=True)
    VQ_model = model.first_stage_model

    csv_path = "/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_embeddings.csv"
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        files_by_label = {}
        for row in reader:
            initial_label = int(row[1])
            if 11 <= initial_label <= 20:
                if initial_label not in files_by_label:
                    files_by_label[initial_label] = []
                files_by_label[initial_label].append(row[0])

    pkl_path = '/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_token_indices.pkl'
    with open(pkl_path, 'rb') as f:
        token_dict = pickle.load(f)

    output_base_path = f"evaluation_results/baseline/model_{args.model}/"
    os.makedirs(output_base_path, exist_ok=True)
    output_file = output_base_path + f'baseline_results.csv'
    
    all_results = {model_name: {metric: [] for metric in ['initial_original_logit', 'initial_target_logit', 
                                                          'initial_original_prob', 'initial_target_prob',
                                                          'final_original_logit', 'final_target_logit', 
                                                          'final_original_prob', 'final_target_prob']}
                   for model_name in pretrained_models}

    top_left = (7, 7)
    bottom_right = (10, 10)

    for initial_label, files in tqdm(files_by_label.items(), desc="Processing labels"):
        target_label = 30 - initial_label
        print(f"Processing label: {initial_label} -> {target_label}")
        
        baseline_csv = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/baseline_statistics/label_{initial_label}.csv"
        baseline_tokens = load_baseline_tokens(baseline_csv)
        
        label_results = process_label(initial_label, target_label, files, token_dict, classification_model, 
                                      pretrained_models, VQ_model, device, baseline_tokens, top_left, bottom_right)
        
        for model_name, model_results in label_results.items():
            for metric, values in model_results.items():
                all_results[model_name][metric].extend(values)

    # Calculate average results
    avg_results = {model_name: {metric: np.mean(values) for metric, values in model_results.items()}
                   for model_name, model_results in all_results.items()}

    # Save results to CSV
    fieldnames = ['model'] + list(next(iter(avg_results.values())).keys())
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model_results in avg_results.items():
            row = {'model': model_name, **model_results}
            writer.writerow(row)

    print("Results saved to", output_file)
    print("Overall average results:")
    for model_name, model_results in avg_results.items():
        print(f"\n{model_name}:")
        for metric, value in model_results.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main(args)