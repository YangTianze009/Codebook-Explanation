import torch
import torch.nn as nn
import torch.optim as optim
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

parser = argparse.ArgumentParser(description='Optimize images for maximum label activation.')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
parser.add_argument('--model', type=int, choices=[1, 2, 3], required=True, help='Classification model to use')
parser.add_argument('--steps', type=int, required=False, default=10000, help='Number of optimization steps')
args = parser.parse_args()

def rescale(x):
    return (x + 1.) / 2.

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

def create_mask(shape, top_left, bottom_right):
    mask = torch.zeros(shape)
    mask[:, :, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = 1
    return mask

def generate_max_activation_image(model, target_label, device, initial_embedding, mask, VQ_model, pretrained_models, initial_label, num_steps=args.steps, lr=0.1, reg=0):
    input_embedding = initial_embedding.clone().detach().requires_grad_(True).to(device)
    mask = mask.to(device)

    optimizer = torch.optim.Adam([input_embedding], lr=lr)
    
    best_results = {model_name: {'step': 0, 'original_logit': 0, 'target_logit': 0} for model_name in pretrained_models}
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        output = model(input_embedding)
        loss = -output[0, target_label] + reg * torch.mean(input_embedding ** 2)
        
        loss.backward()
        
        input_embedding.grad *= mask
        
        optimizer.step()
        
        if step % 500 == 0 or step == num_steps - 1:
            _, _, token_feature_map = decode_input_embedding(input_embedding, VQ_model)
            token_feature_map_img = chw_to_pillow(token_feature_map[0])
            
            preprocessed_img = preprocess_image(token_feature_map_img).to(device)
            
            for model_name, pretrained_model in pretrained_models.items():
                with torch.no_grad():
                    pretrained_output = pretrained_model(preprocessed_img)
                original_logit = pretrained_output[0, initial_label].item()
                target_logit = pretrained_output[0, target_label].item()
                
                if target_logit > best_results[model_name]['target_logit']:
                    best_results[model_name] = {
                        'step': step,
                        'original_logit': original_logit,
                        'target_logit': target_logit
                    }

    _, initial_indices, _ = decode_input_embedding(initial_embedding, VQ_model)
    _, optimized_indices, _ = decode_input_embedding(input_embedding, VQ_model)
    changed_indices = torch.where(initial_indices != optimized_indices)[0]
    # print(f"Changed token indices: {changed_indices}")

    return input_embedding, best_results

def process_label(initial_label, target_label, files, token_dict, classification_model, pretrained_models, VQ_model, device, mask):
    results = []
    for file_name in tqdm(files, desc=f"Processing files for label {initial_label}", leave=False):
        original_image_file = f"/data/ty45972/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test/{file_name}"
        image_array = np.load(original_image_file)
        original_image_tensor = torch.tensor(image_array).unsqueeze(0).to(device)
        
        _, _, original_images = decode_input_embedding(original_image_tensor, VQ_model)
        original_image = chw_to_pillow(original_images[0])
        
        initial_logits = {}
        for model_name, pretrained_model in pretrained_models.items():
            with torch.no_grad():
                preprocessed_img = preprocess_image(original_image).to(device)
                output = pretrained_model(preprocessed_img)
                initial_logits[model_name] = {
                    'original': output[0, initial_label].item(),
                    'target': output[0, target_label].item()
                }
        
        optimized_embedding, best_results = generate_max_activation_image(
            classification_model, target_label, device, original_image_tensor, mask, 
            VQ_model, pretrained_models, initial_label
        )

        file_result = {'file_name': file_name}
        for model_name in pretrained_models:
            initial_original = initial_logits[model_name]['original']
            initial_target = initial_logits[model_name]['target']
            best_original = best_results[model_name]['original_logit']
            best_target = best_results[model_name]['target_logit']
            best_step = best_results[model_name]['step']
            
            file_result[f'{model_name}_original_change'] = best_original - initial_original
            file_result[f'{model_name}_target_change'] = best_target - initial_target
            file_result[f'{model_name}_best_step'] = best_step

        results.append(file_result)

    return results

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    classification_model_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/checkpoints/generated_data/ClassificationNet{args.model}/best_model.pth"
    classification_model = load_classification_model(args.model, classification_model_path, device)
    
    pretrained_models = load_pretrained_models(device)
    
    top_left = (7, 7)
    bottom_right = (10, 10)
    mask = create_mask((1, 256, 16, 16), top_left, bottom_right)

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
            if 10 <= initial_label <= 20 and initial_label != 15:
                if initial_label not in files_by_label:
                    files_by_label[initial_label] = []
                files_by_label[initial_label].append(row[0])

    pkl_path = '/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_token_indices.pkl'
    with open(pkl_path, 'rb') as f:
        token_dict = pickle.load(f)

    output_base_path = f"evaluation_results/optimization/model_{args.model}"
    os.makedirs(output_base_path, exist_ok=True)
    output_file = output_base_path + 'optimization_results_embedding.csv'
    fieldnames = ['initial_label', 'target_label'] + \
                 [f'{model}_{metric}' for model in pretrained_models for metric in ['original_change', 'target_change', 'best_step']]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        all_results = []
        for initial_label, files in tqdm(files_by_label.items(), desc="Processing labels"):
            target_label = 30 - initial_label
            print(f"Processing label: {initial_label} -> {target_label}")
            
            label_results = process_label(initial_label, target_label, files, token_dict, classification_model, pretrained_models, VQ_model, device, mask)
            
            avg_result = {field: np.mean([result[field] for result in label_results]) for field in fieldnames if field in label_results[0]}
            avg_result['initial_label'] = initial_label
            avg_result['target_label'] = target_label
            
            writer.writerow(avg_result)
            all_results.append(avg_result)

        overall_avg = {field: np.mean([result[field] for result in all_results]) for field in fieldnames if field not in ['initial_label', 'target_label']}
        overall_avg['initial_label'] = 'Overall'
        overall_avg['target_label'] = 'Average'
        writer.writerow(overall_avg)

    print("Results saved to", output_file)
    print("Overall average changes:")
    for field, value in overall_avg.items():
        if field not in ['initial_label', 'target_label']:
            print(f"{field}: {value}")

if __name__ == "__main__":
    main(args)