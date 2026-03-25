# Brax Benchmark Guide

本文件说明本仓库中 Brax 论文复现实验入口的用途、运行方式和推荐环境。

## 目标

这部分实验不是沿用 `CEC2013LSGO` 的固定评估次数范式，而是更贴近论文中的机器人控制实验：

- 任务环境：`swimmer`, `hopper`, `reacher`
- 评价目标：最大化策略 reward
- 停止条件：按运行时长

## 核心文件

- [`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)
- [`run_brax_paper.bat`](/D:/my_workspace/demo_1/run_brax_paper.bat)
- [`setup_brax_ubuntu.sh`](/D:/my_workspace/demo_1/setup_brax_ubuntu.sh)
- [`run_brax_paper_ubuntu.sh`](/D:/my_workspace/demo_1/run_brax_paper_ubuntu.sh)
- [`requirements-brax.txt`](/D:/my_workspace/demo_1/requirements-brax.txt)

## 默认配置

- `BRAX_ENVS=swimmer,hopper,reacher`
- `BRAX_TIME_BUDGET_MINUTES=60`
- `BRAX_POP_SIZE=10000`
- `BRAX_MAX_EPISODE_LENGTH=500`
- `BRAX_NUM_EPISODES=1`
- `BRAX_HIDDEN_DIMS=32,32`
- `BRAX_LOG_INTERVAL_SECONDS=60`
- `BRAX_SEED=42`
- `BRAX_SAVE_HTML=0`

## Ubuntu 推荐运行方式

```bash
bash ./setup_brax_ubuntu.sh
BRAX_ENVS=swimmer BRAX_TIME_BUDGET_MINUTES=5 bash ./run_brax_paper_ubuntu.sh
bash ./run_brax_paper_ubuntu.sh
```

## Windows 说明

Windows 下已经提供：

- [`run_brax_paper.bat`](/D:/my_workspace/demo_1/run_brax_paper.bat)

但当前 `Windows + Python 3.13` 并不是理想的 Brax/JAX 运行组合。
如果目标是稳定复现论文中的 Brax 实验，更推荐：

- Ubuntu
- Python 3.10 到 3.12
- CUDA JAX

## 输出文件

每次运行会在 `save_dir/brax/ISTRATDE_<timestamp>/` 下生成：

- `summary_all.json`
- 每个环境一个子目录

每个环境子目录里通常包括：

- `summary.json`
- `reward_history.csv`
- `reward_curve.png`
- `reward_curve.pdf`
- `best_solution.pt`
- `best_policy.html`（开启 `BRAX_SAVE_HTML=1` 时）
