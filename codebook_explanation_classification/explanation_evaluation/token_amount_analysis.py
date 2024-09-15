import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import pickle
import csv
import argparse
from tqdm import tqdm
import itertools

def load_top_tokens(csv_path, top_n, token_number):
    target_token_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        in_top_n_section = False
        current_row = 0

        for row in reader:
            if f"Top {top_n} Tokens" in row:
                in_top_n_section = True
                next(reader)
                current_row = 0
                continue

            if in_top_n_section:
                if "Top" in row[0] or current_row == token_number:
                    break
                token = int(row[0])
                target_token_list.append(token)
                current_row += 1

    return target_token_list

def load_top_tokens_baseline(input_csv, top_n):
    tokens = []
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            token = int(row['Token'])
            count = int(row['Count'])
            tokens.append((token, count))
    
    tokens.sort(key=lambda x: x[1], reverse=True)
    top_n_tokens = [token for token, _ in tokens[:top_n]]
    
    return top_n_tokens

def count_masked_tokens(token_list, target_token_list):
    return sum(1 for token in token_list if token in target_token_list)

def main(args):
    # Load token dictionary
    with open('/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_token_indices.pkl', 'rb') as f:
        token_dict = pickle.load(f)

    # Define paths
    test_csv = "/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_embeddings.csv"

    # Define combinations of top_n and token_num
    top_n_values = [1, 5, 10, 20]
    token_num_values = [10, 20, 30, 40, 50, 100]
    combinations = list(itertools.product(top_n_values, token_num_values))

    # Prepare output CSV
    output_csv = "token_amount_analysis.csv"
    evaluation_result_path = "evaluation_results/"
    os.makedirs(evaluation_result_path, exist_ok=True)
    
    with open(os.path.join(evaluation_result_path, output_csv), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Top_N', 'Token_Num', 'Avg_Masked_Tokens_Target', 'Avg_Masked_Tokens_Baseline'])

        for top_n, token_num in tqdm(combinations, desc="Processing combinations"):
            total_masked_tokens_target = 0
            total_masked_tokens_baseline = 0
            total_images = 0

            # Read test CSV and process each image
            with open(test_csv, 'r', encoding='utf-8') as test_csvfile:
                reader = csv.reader(test_csvfile)
                next(reader)  # Skip header
                for row in reader:
                    filename = row[0]
                    label = int(row[1])

                    token_list = token_dict.get(filename)
                    if token_list is None:
                        continue

                    csv_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/generated_data/label/Net1/label_activation_statistics/label_{label}.csv"
                    baseline_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/baseline_statistics/label_{label}.csv"

                    # Load tokens
                    target_token_list = load_top_tokens(csv_path, top_n, token_num)
                    target_token_list_baseline = load_top_tokens_baseline(baseline_path, token_num)

                    masked_tokens_target = count_masked_tokens(token_list, target_token_list)
                    masked_tokens_baseline = count_masked_tokens(token_list, target_token_list_baseline)
                    
                    total_masked_tokens_target += masked_tokens_target
                    total_masked_tokens_baseline += masked_tokens_baseline
                    total_images += 1

            if total_images > 0:
                avg_masked_tokens_target = total_masked_tokens_target / total_images
                avg_masked_tokens_baseline = total_masked_tokens_baseline / total_images
                
                # Write to CSV file
                writer.writerow([top_n, token_num, avg_masked_tokens_target, avg_masked_tokens_baseline])
                
                # Flush file buffer to ensure data is written to disk
                csvfile.flush()

    print(f"Analysis complete. Results saved to {os.path.join(evaluation_result_path, output_csv)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token amount analysis for different top_n and token_num combinations.")
    parser.add_argument('--gpu', type=int, default=None, help="Specify GPU to use (e.g., 0 or 1). If not specified, will use any available GPU or CPU.")
    args = parser.parse_args()
    
    main(args)