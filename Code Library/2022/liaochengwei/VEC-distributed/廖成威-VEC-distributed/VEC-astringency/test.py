import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 8 * 4, 128)  # 8x4 is the size after the convolution
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Assuming 10 actions in the output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16 * 8 * 4)  # Flatten the output for the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # Output softmax probabilities for actions

# 创建模型实例
model = CustomNet()

# 创建输入数据，将二维列表转换为适当的形状
state_list = [
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
    [37, 38, 39, 40, 41, 42],
    [43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54],
    [55, 56, 57, 58, 59, 60]
]
state_array = torch.tensor(state_list, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# 前向传播
action_probs = model(state_array)

# 输出动作概率
print(action_probs)
