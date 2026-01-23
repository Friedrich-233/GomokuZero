
# ST449 Artificial Intelligence Group Coursework

The game of Gomoku arose around the mid-1700s, and has consistently been popular in the area of chess.  Since 2016, the AlghaGo has dominated Go, and so have all other chess games. 
Even though it is not a state-of-the-art topic for AI with constraint satisfaction problems, it is still worth investigating how various methodologies could be applied in the game and their efficiency. 

## Menu

- `gomoku/`：
- `models/`：

## Improvement

- Squeeze-and-Excitation (SE) Mechanism.


## Deploy

Python 3.12+。

```bash
python -m pip install -e .
```

## Quick Start

Local GUI（Pygame）：

```bash
python gomoku/gui.py
```

```bash
uvicorn gomoku.app:app --reload
```

Main Variables：

- `GOMOKU_MODEL_PATH`： `models/gomoku_zero_9_lab_4/policy_step_30000.pth`
- `GOMOKU_MCTS_ITERS`：MCTS Iters（Default `400`）
- `GOMOKU_MCTS_PUCT`：PUCT Constant（Default `2.0`）
- `GOMOKU_AI_WORKERS`：AI Workers（Default `2`）


## Tournament 

```bash
python3 -m gomoku.arena_comparison_pro
```
