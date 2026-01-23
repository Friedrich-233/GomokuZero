import torch
import os
import rich
from rich.table import Table
from gomoku.player import arena_parallel
from gomoku.policy import ZeroPolicy  

import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
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
    def __init__(self, channels, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + res)

class UniversalPolicy(nn.Module):
    
    def __init__(self, board_size=9, is_attn=True):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.res_blocks = nn.ModuleList([ResBlock(32, use_se=is_attn) for _ in range(2)])
        self.policy_conv = nn.Conv2d(32, 2, kernel_size=1); self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.value_conv = nn.Conv2d(32, 1, kernel_size=1); self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64); self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() == 2: x = x.view(batch_size, 3, self.board_size, self.board_size)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks: x = block(x)
        p = self.policy_fc(F.relu(self.policy_bn(self.policy_conv(x))).view(batch_size, -1))
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(F.relu(self.value_bn(self.value_conv(x))).view(batch_size, -1)))))
        return p, v

if __name__ == "__main__":
    base_path = "models/policy_step_50000.pth"
    attn_path = "models/policy_step_attention_50000.pth"
    
    BOARD_SIZE = 9
    ITERS = 400
    GAMES_TOTAL = 2  

    rich.print("attention vs baseline")

    m_base = UniversalPolicy(BOARD_SIZE, is_attn=False)
    m_base.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True))
    
    m_attn = UniversalPolicy(BOARD_SIZE, is_attn=True)
    m_attn.load_state_dict(torch.load(attn_path, map_location="cpu", weights_only=True))
    
    m_base.eval(); m_attn.eval()

    rich.print(f"\n[yellow]Round 1: Attention(P1) vs Baseline(P2) | {GAMES_TOTAL//2} Games[/yellow]")
    res1 = arena_parallel(m_attn, m_base, board_size=BOARD_SIZE, num_cpus=16, games=GAMES_TOTAL//2, itermax=ITERS, eager=True)

    rich.print(f"\n[yellow]Round 2: Baseline(P1) vs Attention(P2) | {GAMES_TOTAL//2} Games[/yellow]")
    res2 = arena_parallel(m_base, m_attn, board_size=BOARD_SIZE, num_cpus=16, games=GAMES_TOTAL//2, itermax=ITERS, eager=True)

    attn_wins = int(res1['player1_win_rate'] * (GAMES_TOTAL//2)) + int(res2['player2_win_rate'] * (GAMES_TOTAL//2))
    base_wins = int(res1['player2_win_rate'] * (GAMES_TOTAL//2)) + int(res2['player1_win_rate'] * (GAMES_TOTAL//2))
    
    table = Table(title="results")
    table.add_column("Model", style="magenta")
    table.add_column("Total Win", justify="center", style="green")
    table.add_column("Win Rate", justify="right", style="bold")
    
    table.add_row("Attention (SE 50000 step)", str(attn_wins), f"{(attn_wins/GAMES_TOTAL):.1%}")
    table.add_row("Baseline (Std 50000 step)", str(base_wins), f"{(base_wins/GAMES_TOTAL):.1%}")
    
    rich.print("\n", table)