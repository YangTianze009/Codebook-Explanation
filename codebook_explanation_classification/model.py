import torch.nn as nn
import torch
import math    


class ClassificationNet1(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNet1, self).__init__()
        
        self.conv1_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(512)
        self.conv1_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(512)
        self.conv1_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_4 = nn.BatchNorm2d(512)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(512)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(512)
        self.conv2_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(512)
        self.conv2_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_4 = nn.BatchNorm2d(512)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn_fc2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.relu(self.bn1_3(self.conv1_3(x)))
        x = self.relu(self.bn1_4(self.conv1_4(x)))
        x = self.maxpool1(x)
        
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        x = self.relu(self.bn2_4(self.conv2_4(x)))
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x


class ClassificationNet2(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNet2, self).__init__()
        
        self.conv1_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(512)
        self.conv1_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(512)
        self.conv1_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_4 = nn.BatchNorm2d(512)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(512)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(512)
        self.conv2_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(512)
        self.conv2_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_4 = nn.BatchNorm2d(512)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # New convolutional layers added after the second max pool
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)
        
        # Adjusted fully connected layers based on the new feature map size
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.bn_fc2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.relu(self.bn1_3(self.conv1_3(x)))
        x = self.relu(self.bn1_4(self.conv1_4(x)))
        x = self.maxpool1(x)
        
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        x = self.relu(self.bn2_4(self.conv2_4(x)))
        x = self.maxpool2(x)
        
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
# 定义 SE 模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        out = x.mean(dim=[2, 3])  # Global Average Pooling
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out)).view(batch_size, num_channels, 1, 1)
        return x * out

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 增加 SE Block
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.se(out)  # 添加 SE Block
        out = self.relu(out)
        return out

# 定义分类网络
class ClassificationNet3(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNet3, self).__init__()
        
        # 保持 ResNet 特性，增加残差块数量
        self.layer1 = self._make_layer(256, 512, 3)
        self.layer2 = self._make_layer(512, 512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 2048)  # 减少神经元数到 2048
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)  # 适当减少 Dropout
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈神经网络
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)

        return x

class ClassificationNet4(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=1024, num_layers=3, num_classes=1000):
        super(ClassificationNet4, self).__init__()
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = 16 * 16  # 输入已经是16x16个patch，每个patch的维度是256
        self.pos_embed = self._build_sinusoidal_embeddings(num_patches + 1, embed_dim)  # 使用正弦-余弦位置编码

        # 创建多个Transformer编码层
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(65792, num_classes)

    def _build_sinusoidal_embeddings(self, num_positions, embed_dim):
        """构建正弦-余弦位置编码"""
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_positions, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        # 输入展平后，每个patch的维度是256，patch数目是16*16
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        # print("x", x.shape)

        # Add class token and position embedding
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_dim]
        x = x + self.pos_embed[:, :x.size(1), :].to(x.device)  # [batch_size, num_patches + 1, embed_dim]
        # Transformer encoder
        for layer in self.encoder:
            x = layer(x)

        # Classification head
        x = self.norm(x)
        # print("x", x.shape)
        cls_token_final = x.flatten(1)
        #cls_token_final = x[:, 0]  # 取出CLS Token
        # print("cls_token_final", cls_token_final.shape)
        x = self.fc(cls_token_final)  # [batch_size, num_classes]

        return x



if __name__ == "__main__":
    model = ClassificationNet3(1000)
    print("model", model)
    input_tensor = torch.randn(32, 256, 16, 16)
    output = model(input_tensor)
    print(output.shape)