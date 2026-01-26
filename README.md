
# ST449 Artificial Intelligence Group Coursework

This work investigates two complementary approaches to Gomoku: enhanced Alpha-Beta search with transposition tables and move ordering, and neural Monte Carlo Tree Search augmented with Squeeze-and-Excitation (SE) attention. By integrating lightweight channel attention into residual blocks, the policy network dynamically focuses on tactically relevant features during self-play. Experimental results on a $9 \times 9$ board demonstrate that the SE-Attention model outperforms the standard ResNet baseline in head-to-head competition while incurring minimal computational overhead. 

## Menu

- `gomoku/`：Core implementation of the Gomoku AI, including the game environment, MCTS logic, policy/value networks, and training pipeline.
- `models/`：Saved model checkpoints and trained parameters produced during self-play training.

## Improvement

- **Squeeze-and-Excitation (SE) Mechanism**  
  We integrate lightweight SE attention into residual blocks to perform channel-wise feature reweighting.  
  By adaptively emphasising tactically relevant feature channels (e.g. threat and blocking patterns), the policy network can better focus on critical board information during self-play, while introducing only minimal computational overhead.


## Deploy

Python 3.12+。

```bash
python -m pip install -e .
```

## Local GUI（Pygame）

A simply gui allow you to play with the models：

```bash
python gomoku/gui.py
```
## Tournament 

Run a tournament script that lets four different Gomoku models play head-to-head matches for performance comparison.

```bash
python3 -m gomoku.arena_comparison_pro
```

## Main Variables

- `GOMOKU_MODEL_PATH`： `models/policy_step_50000.pth`
- `GOMOKU_MCTS_ITERS`：MCTS Iters（Default `400`）
- `GOMOKU_MCTS_PUCT`：PUCT Constant（Default `2.0`）
- `GOMOKU_AI_WORKERS`：AI Workers（Default `2`）


