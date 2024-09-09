import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptModel1(nn.Module):
    def __init__(self, num_concepts):
        super(ConceptModel1, self).__init__()
        
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
        
        self.fc1 = nn.Linear(512 * 4 * 4, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_concepts)
        
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
        x = torch.sigmoid(x)
        return x
    

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class ConceptModel2(nn.Module):
    def __init__(self, num_concepts):
        super(ConceptModel2, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(256, 64, kernel_size=3, stride=1)
        self.Mixed_5b = InceptionA(64, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.fc = nn.Linear(288 * 4 * 4, 512)  # Adjusted to match output size
        self.out_fc = nn.Linear(512, num_concepts)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = F.adaptive_avg_pool2d(x, (4, 4))  # Adjust pooling to match input size
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.out_fc(x)
        x = torch.sigmoid(x)
        return x

    
# Example usage
if __name__ == "__main__":
    num_concepts = 50  # Example number of concepts
    model = ConceptModel2(num_concepts=num_concepts)
    example_input = torch.randn(1, 256, 16, 16)  # Example input
    output = model(example_input)
    print(output)

