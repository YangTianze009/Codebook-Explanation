#!/bin/bash


# 运行 ViT-B/32 模型，top_n=20，token_num=50
echo "Running analysis for ViT-B/16, top_n=10, token_num=100"
python logits_comparision.py --model vit_b_16 --top_n 10 --token_num 100 --gpu 0
echo "----------------------------------------"

# 运行 ViT-B/32 模型，top_n=20，token_num=100
echo "Running analysis for ViT-B/16, top_n=20, token_num=100"
python logits_comparision.py --model vit_b_16 --top_n 20 --token_num 100 --gpu 0
echo "----------------------------------------"

echo "All analyses completed."