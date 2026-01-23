import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ray
import os
import matplotlib.pyplot as plt
import seaborn as sns
from rich.table import Table
from rich.console import Console
from gomoku.player import ZeroMCTSPlayer, play_one_game
from gomoku.gomoku_env import GomokuEnv
from gomoku.alpha_beta_baseline import AlphaBeta
from gomoku.alpha_beta_plus import AlphaBetaCM

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

@ray.remote
class TourneyWorker:
    def __init__(self, board_size):
        self.board_size = board_size

    def fight(self, p1, p2, p1_name, p2_name, itermax=400):
        winner, _ = play_one_game(
            p1, p2, 
            board_size=self.board_size, 
            itermax=itermax, 
            eager=False, 
            use_dirichlet=True
        )
        return {"p1": p1_name, "p2": p2_name, "winner": winner}

def run_5w_king_tournament():
    board_size = 9
    path_attn = "models/policy_step_attention_50000.pth"
    path_base = "models/policy_step_50000.pth"
    
    pol_attn = UniversalPolicy(board_size, is_attn=True)
    pol_attn.load_state_dict(torch.load(path_attn, map_location="cpu"))
    
    pol_base = UniversalPolicy(board_size, is_attn=False)
    pol_base.load_state_dict(torch.load(path_base, map_location="cpu"))
    
    ab_base = AlphaBeta(search_depth=3)
    ab_plus = AlphaBetaCM(search_depth=3)
    

    ab_base.policy = None 
    ab_plus.policy = None

    players = {
        "Attention_Zero_50k": ZeroMCTSPlayer(pol_attn), 
        "Baseline_Zero_50k": ZeroMCTSPlayer(pol_base),
        "AlphaBeta_Baseline": ab_base, 
        "AlphaBeta_Plus": ab_plus    
    }
    
    names = list(players.keys())
    if not ray.is_initialized(): ray.init(num_cpus=16)
    
    worker = TourneyWorker.remote(board_size)
    tasks = []
    GAMES_PER_PAIR = 10 

    print(f"competition begins!")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1_n, p2_n = names[i], names[j]
            for _ in range(GAMES_PER_PAIR):
                tasks.append(worker.fight.remote(players[p1_n], players[p2_n], p1_n, p2_n))
                tasks.append(worker.fight.remote(players[p2_n], players[p1_n], p2_n, p1_n))

    results = ray.get(tasks)
    
    stats = {name: {"win": 0, "draw": 0, "loss": 0} for name in names}
    win_matrix = np.zeros((len(names), len(names)))
    
    for res in results:
        p1, p2, winner = res["p1"], res["p2"], res["winner"]
        idx1, idx2 = names.index(p1), names.index(p2)
        if winner == 0:
            stats[p1]["draw"] += 1; stats[p2]["draw"] += 1
        elif winner == 1: 
            stats[p1]["win"] += 1; stats[p2]["loss"] += 1
            win_matrix[idx1, idx2] += 1
        else: 
            stats[p2]["win"] += 1; stats[p1]["loss"] += 1
            win_matrix[idx2, idx1] += 1

    table = Table(title="Ranking")
    table.add_column("Candidate", style="cyan"); table.add_column("Total Score", style="bold yellow")
    table.add_column("Win-Draw-Loss")
    
    sorted_names = sorted(names, key=lambda x: stats[x]["win"] + stats[x]["draw"]*0.5, reverse=True)
    for n in sorted_names:
        s = stats[n]
        score = s["win"] + s["draw"]*0.5
        table.add_row(n, f"{score:.1f}", f"{s['win']}-{s['draw']}-{s['loss']}")
    
    Console().print(table)
    return names, win_matrix

def plot_heatmap(names, matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, xticklabels=names, yticklabels=names, cmap="rocket_r")
    plt.title("competition matrix")
    plt.savefig("50k_king_matrix.png")
    plt.show()

if __name__ == "__main__":
    names, matrix = run_5w_king_tournament()
    plot_heatmap(names, matrix)