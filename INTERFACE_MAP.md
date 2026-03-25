# iStratDE Interface Map

本文件用于说明本仓库目前真正暴露出来的 iStratDE 接口，以及这些接口如何映射到上游 `istratde`。

## 1. 外部统一接口

### 适配器入口

文件：

- [`MMES/istratde_optimizer.py`](/D:/my_workspace/demo_1/MMES/istratde_optimizer.py)

对外接口：

```python
optimizer = IStratDEOptimizer(problem, options)
results = optimizer.optimize()
```

### `problem` 结构

```python
problem = {
    "fitness_function": callable,
    "ndim_problem": int,
    "lower_boundary": np.ndarray,
    "upper_boundary": np.ndarray,
}
```

### `options` 结构

常用字段：

```python
options = {
    "max_function_evaluations": int,
    "n_individuals": int,
    "seed_rng": int,
    "backend": "torch" | "jax",
    "mean": np.ndarray | None,
    "sigma": float | None,
    "device": "cuda" | "cpu" | None,
    "verbose": int,
}
```

### `results` 结构

常见返回项：

```python
results = {
    "best_so_far_x": np.ndarray,
    "best_so_far_y": float,
    "n_function_evaluations": int,
    "runtime": float,
    "fitness": list | np.ndarray,
    "profiling": dict,
    "backend": str,
}
```

## 2. 本仓库内部到上游 iStratDE 的映射

### CEC / 通用黑盒优化路径

本仓库接口：

- [`MMES/istratde_optimizer.py`](/D:/my_workspace/demo_1/MMES/istratde_optimizer.py)

使用的上游接口：

- `istratde.algorithms.pytorch.istratde.IStratDE`
- `istratde.util.workflows.StdWorkflow`
- `istratde.util.workflows.EvalMonitor`

本仓库做的额外工作：

- 把本地 `problem/options` 包装成 iStratDE 可接受的 workflow
- 补齐进度输出
- 补齐 profiling
- 把结果转回本地实验框架熟悉的 `results` 字典

### Brax 路径

本仓库接口：

- [`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)

使用的上游接口：

- `istratde.algorithms.pytorch.istratde.IStratDE`
- `istratde.problems.torch.brax.BraxProblem`
- `istratde.util.workflows.StdWorkflow`
- `istratde.util.workflows.EvalMonitor`
- `istratde.util.ParamsAndVector`

本仓库做的额外工作：

- 统一环境变量配置
- 改成按运行时长停止
- 记录 reward history
- 生成 CSV、JSON、图表和最佳策略文件

## 3. 实验入口接口

### `test.py`

文件：

- [`test.py`](/D:/my_workspace/demo_1/test.py)

主要环境变量：

- `DEMO_OPTIMIZER`
- `ISTRATDE_BACKEND`
- `POP_SIZE`
- `MAX_FES`
- `CYCLE_NUM`
- `FUN_ID_START`
- `FUN_ID_END`
- `VERBOSE_EVERY`

适用场景：

- `CEC2013LSGO`
- `MMES / ISTRATDE` 对照

### `brax_paper_benchmark.py`

文件：

- [`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)

主要环境变量：

- `BRAX_ENVS`
- `BRAX_TIME_BUDGET_MINUTES`
- `BRAX_POP_SIZE`
- `BRAX_MAX_EPISODE_LENGTH`
- `BRAX_NUM_EPISODES`
- `BRAX_HIDDEN_DIMS`
- `BRAX_LOG_INTERVAL_SECONDS`
- `BRAX_SEED`
- `BRAX_SAVE_HTML`
- `BRAX_BACKEND`

适用场景：

- Brax 机器人控制论文复现

## 4. 服务器接入时建议保留的核心文件

如果后续迁移到实验室服务器，建议至少保留：

- [`MMES/istratde_optimizer.py`](/D:/my_workspace/demo_1/MMES/istratde_optimizer.py)
- [`MMES/optimizer.py`](/D:/my_workspace/demo_1/MMES/optimizer.py)
- [`test.py`](/D:/my_workspace/demo_1/test.py)
- [`utils.py`](/D:/my_workspace/demo_1/utils.py)
- [`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)
- [`setup_brax_ubuntu.sh`](/D:/my_workspace/demo_1/setup_brax_ubuntu.sh)
- [`run_brax_paper_ubuntu.sh`](/D:/my_workspace/demo_1/run_brax_paper_ubuntu.sh)
- [`requirements.txt`](/D:/my_workspace/demo_1/requirements.txt)
- [`requirements-brax.txt`](/D:/my_workspace/demo_1/requirements-brax.txt)

此外还需要：

- `cec2013lsgo/`
- `istratde-main/src/istratde/`

## 5. 当前边界

### CEC 路径

- 算法：GPU
- benchmark 评估：CPU

### Brax 路径

- 已补齐论文复现实验入口
- 更推荐在 Ubuntu 服务器上运行
- 推荐环境：`Python 3.10~3.12 + CUDA + JAX + Brax`
