import torch.nn as nn
import torch
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        

        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)
        
        out += residual
        out = F.relu(out)
        return out

class ZeroPolicy(nn.Module):
    def __init__(self, board_size, num_blocks = 2):
        super(ZeroPolicy, self).__init__()
        self.board_size = board_size
        self.channel_size = 3  
        
        self.conv1 = nn.Conv2d(self.channel_size, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res_blocks = nn.ModuleList([
            ResBlock(32, 32) for _ in range(num_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(32, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        self.value_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.view(batch_size, self.channel_size, self.board_size, self.board_size)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)
        
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(batch_size, -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)  
        
        return policy, value

if __name__ == "__main__":

    model = ZeroPolicy(9, num_blocks=2)
    p, v = model(torch.randn(1, 3 * 9 * 9))
    print(f"Policy shape: {p.shape}") 
    print(f"Value shape: {v.shape}")  