import torch
import torchvision
from torchvision import transforms
from PIL import Image
import glob
import urllib.request

# 1. 加载预训练的ViT模型
# model = torchvision.models.vit_b_16(pretrained=True)
# model = torchvision.models.vit_b_32(pretrained=True)
# model = torchvision.models.resnet18(pretrained=True)
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# 2. 定义图像预处理步骤（不调整尺寸）
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    )
])


# 3. 指定您的图片路径（请修改为您的实际路径）
image_paths = glob.glob("/data2/ty45972_data2/taming-transformers/datasets/imagenet_VQGAN_generated/80/000000.png")  # 请修改路径和文件扩展名

# 4. 对每张图片进行分类预测
for image_path in image_paths:
    # 加载并预处理图片
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 添加批次维度

    # 将数据和模型移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    model.to(device)

    # 禁用梯度计算，进行前向传播
    with torch.no_grad():
        output = model(input_batch)

    # 计算概率
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 获取预测的前5个类别
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # 输出结果
    print(f"图片: {image_path}")
    for i in range(top5_prob.size(0)):
        print(f"类别: {top5_catid[i]}, 概率: {top5_prob[i].item():.4f}")
    print("-" * 50)
