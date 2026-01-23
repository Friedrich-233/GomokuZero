import os
from collections import deque
from gomoku.evaluate import (
    evaluate_validation_samples,
    load_validation_samples,
    resolve_validation_path,
)
from gomoku.player import arena_parallel
from gomoku.worker import gather_selfplay_games, get_symmetric_data
import random
from gomoku.policy import ZeroPolicy # 这里会自动加载你刚才修改的带 SEBlock 的类
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn.functional as F
import rich
import tqdm
import numpy as np
import ray

# --- 实验配置 (论文实验组) ---
board_size = 9
lr = 1e-3
save_per_steps = 10000 # 建议缩短保存间隔，方便收集对战样本
cpus = 16
device = "cuda"
seed = 42

# 更改实验室名称，用于论文对比实验
lab_name = "gomoku_zero_9_se_attention" 
comment = "SE-ResNet architecture, 2 blocks, 32 channels, dirichlet epsilon=0.15"

batch_size = 256
threshold = 0.2
alpha = 2.0
itermax = 400
validation_eval_step = 1000
validation_top_k = 5
validation_path = None

# 根据棋盘大小自动调整超参数
if board_size == 9:
    steps = 50009
    buffer_size = 60000
    self_play_per_steps = 250
    self_play_num = 32
    eval_steps = 5000 # 每 5000 步进行一次模型自我进化评测
    games_per_worker = self_play_num // cpus
    num_workers = cpus

def train(policy: ZeroPolicy, optimizor, replay_buffer):
    writer = SummaryWriter(f"runs/{lab_name}", comment=comment)

    exclude_list = [
        ".git", "__pycache__", "*.pyc", "*.o", ".idea", ".vscode",
        "checkpoints/", "runs/", ".venv/", "models/"
    ]

    ray.init(
        num_cpus=cpus,
        runtime_env={"excludes": exclude_list, "working_dir": None},
    )

    # 论文推荐：CosineAnnealingLR 能更好地帮助注意力权重在后期平滑收敛
    scheduler = CosineAnnealingLR(optimizor, T_max=steps, eta_min=1e-4)

    # 用于竞技场评测的当前最佳模型
    best_policy = ZeroPolicy(board_size=board_size)
    best_policy.load_state_dict(policy.state_dict())

    update_count = 0

    for step in tqdm.tqdm(range(steps)):
        policy.train()
        
        # 1. 自对弈生成样本
        if step % self_play_per_steps == 0:
            with torch.no_grad():
                generate_model = policy
                generate_model.eval()
                
                games = gather_selfplay_games(
                    generate_model,
                    "cpu",
                    board_size=board_size,
                    itermax=itermax,
                    games_per_worker=games_per_worker,
                    num_workers=num_workers,
                )
                
                for game in games:
                    for i in range(len(game["states"])):
                        # 利用对称性扩充 8 倍样本，加速注意力机制学习
                        augmented_samples = get_symmetric_data(
                            game["states"][i], game["probs"][i], board_size=board_size
                        )
                        for state, pi in augmented_samples:
                            replay_buffer.append((state, pi, game["rewards"][i]))
                rich.print(f"[green]Step {step}: Self play generated {len(games)} games[/green]")

        # 2. 竞技场评测：新模型 vs 最佳模型
        if step != 0 and step % eval_steps == 0:
            policy.eval()
            best_policy.eval()

            # 深拷贝模型到 CPU 进行并行对战
            policy_cpu_copy = ZeroPolicy(board_size=board_size).to("cpu")
            policy_cpu_copy.load_state_dict(policy.state_dict())

            best_policy_cpu_copy = ZeroPolicy(board_size=board_size).to("cpu")
            best_policy_cpu_copy.load_state_dict(best_policy.state_dict())

            r = arena_parallel(
                policy_cpu_copy,
                best_policy_cpu_copy,
                games=48,
                board_size=board_size,
                num_cpus=cpus,
                eager=False,
                itermax=itermax,
            )

            win_rate = r["player1_win_rate"]
            # 只有胜率显著提升时，才更新最佳模型 (论文中可作为模型进化证据)
            if win_rate >= 0.55:
                best_policy.load_state_dict(policy.state_dict())
                update_count += 1
                rich.print(f"[bold yellow]New best model updated! Win rate: {win_rate:.2%}[/bold yellow]")

            writer.add_scalar("Train/win-rate", win_rate, step)
            writer.add_scalar("Train/update-count", update_count, step)

        # 3. 核心训练逻辑
        if len(replay_buffer) < batch_size:
            continue

        policy.train()
        batch = random.sample(replay_buffer, batch_size)
        states, probs, rewards = zip(*batch)
        
        states = torch.from_numpy(np.array(states)).float().to(device)
        probs = torch.from_numpy(np.array(probs)).float().to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        optimizor.zero_grad()
        logits, value = policy(states)

        # 损失函数：均方误差(胜率) + 交叉熵(策略)
        mse = F.mse_loss(value.squeeze(), rewards, reduction="mean")
        log_probs = F.log_softmax(logits, dim=-1)
        cse = -torch.sum(probs * log_probs, dim=1).mean()

        loss = alpha * mse + cse
        loss.backward()
        optimizor.step()
        scheduler.step()

        # 记录训练数据到 TensorBoard
        writer.add_scalar("Train/loss", loss.item(), step)
        writer.add_scalar("Train/mse", mse.item(), step)
        writer.add_scalar("Train/lr", optimizor.param_groups[0]["lr"], step)

        if step % 100 == 0:
            rich.print(f"step: {step}, loss: {loss.item():.4f}, mse: {mse.item():.4f}, lr: {optimizor.param_groups[0]['lr']:.6f}")

        # 4. 定期保存权重
        if step != 0 and step % save_per_steps == 0:
            save_path = f"models/{lab_name}/policy_step_{step}.pth"
            torch.save(policy.state_dict(), save_path)
            rich.print(f"[bold cyan]Saved SE-Attention model at {save_path}[/bold cyan]")

if __name__ == "__main__":
    if not os.path.exists(f"models/{lab_name}"):
        os.makedirs(f"models/{lab_name}")

    random.seed(seed)
    buffer = deque(maxlen=buffer_size)

    # 实例化新的 SE-ResNet 策略网络
    policy = ZeroPolicy(board_size=board_size)
    policy.to(device)
    
    # 权重衰减 (L2 正则化) 有助于注意力权重的泛化
    optimizor = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
    
    train(policy, optimizor, buffer)
