import csv
from collections import defaultdict, Counter
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

# 输入CSV文件的路径
input_csv = '/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/train_embeddings.csv'  # 请将此处替换为您的CSV文件路径

# 初始化一个字典来存储每个label对应的token计数
label_token_counts = defaultdict(Counter)

# 首先计算文件的总行数，以便在进度条中显示
with open(input_csv, 'r', encoding='utf-8') as csvfile:
    total_lines = sum(1 for _ in csvfile) - 1  # 减去表头
    csvfile.seek(0)  # 返回文件开头
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头

    # 使用 tqdm 包装 reader，以显示进度条
    for row in tqdm(reader, total=total_lines, desc='Processing rows'):
        embedding = row[0]
        label = row[1]
        indices_str = row[2]

        # 解析indices字符串为整数列表
        indices_list = indices_str.strip('[]').split(', ')
        indices_list = [int(token) for token in indices_list if token]

        # 更新对应label的token计数
        label_token_counts[label].update(indices_list)

# 对每个label，获取出现次数最多的前100个token，并保存到新的CSV文件
for label, token_counter in tqdm(label_token_counts.items(), desc='Processing labels'):
    # 获取前100个最常见的token
    top_tokens = token_counter.most_common(100)

    # 创建输出CSV文件名
    output_csv = f'/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/baseline_statistics/label_{label}.csv'

    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Token', 'Count'])
        # 写入token和对应的计数
        for token, count in top_tokens:
            writer.writerow([token, count])

    print(f'Label {label}: Top tokens saved to {output_csv}')
