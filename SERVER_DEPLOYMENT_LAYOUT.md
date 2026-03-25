# 实验室服务器部署版目录说明

本文件用于说明如何把当前仓库整理并部署到实验室 Ubuntu GPU 服务器上。

目标不是把所有本地文件原样复制上去，而是保留核心源码、单独管理环境与输出目录，让后续批量实验、日志管理和多人协作更清晰。

## 1. 推荐部署目标

推荐部署环境：

- Ubuntu 22.04 / 24.04
- Python 3.10 到 3.12
- NVIDIA 驱动正常
- CUDA 可用
- `nvidia-smi` 正常

推荐用途分为两类：

- `CEC2013LSGO` 黑盒优化实验
- `Brax` 机器人控制论文复现实验

## 2. 推荐服务器目录结构

建议在服务器上按下面结构组织：

```text
~/projects/
`-- istratde-interface/
    |-- MMES/
    |-- cec2013lsgo/
    |-- istratde-main/
    |-- test.py
    |-- utils.py
    |-- brax_paper_benchmark.py
    |-- compare_results.py
    |-- run_test_istratde.bat
    |-- run_compare_f15.bat
    |-- run_brax_paper.bat
    |-- setup_brax_ubuntu.sh
    |-- run_brax_paper_ubuntu.sh
    |-- requirements.txt
    |-- requirements-brax.txt
    |-- README.md
    |-- README_BRAX.md
    |-- INTERFACE_MAP.md
    |-- SERVER_DEPLOYMENT_LAYOUT.md
    |-- .venv-cec/              # 服务器本地创建，不上传
    |-- .venv-brax/             # 服务器本地创建，不上传
    |-- save_dir/               # 实验输出，不上传
    |-- logs/                   # 建议新增，保存 nohup/slurm 日志
    `-- tmp/                    # 临时目录，不上传
```

如果实验室服务器空间充足，也可以把环境和输出放到统一的实验目录中：

```text
~/experiments/istratde-interface/
|-- repo/                      # git clone 下来的源码
|-- envs/
|   |-- cec/
|   `-- brax/
|-- outputs/
|   |-- cec/
|   `-- brax/
`-- logs/
```

这种方式更适合长期维护，也更适合多次重复实验。

## 3. 建议上传到服务器的核心文件

建议保留并同步到服务器的核心文件：

- [`MMES/istratde_optimizer.py`](/D:/my_workspace/demo_1/MMES/istratde_optimizer.py)
- [`MMES/optimizer.py`](/D:/my_workspace/demo_1/MMES/optimizer.py)
- [`MMES/mmes.py`](/D:/my_workspace/demo_1/MMES/mmes.py)
- [`cec2013lsgo`](/D:/my_workspace/demo_1/cec2013lsgo)
- [`istratde-main/src/istratde`](/D:/my_workspace/demo_1/istratde-main/src/istratde)
- [`test.py`](/D:/my_workspace/demo_1/test.py)
- [`utils.py`](/D:/my_workspace/demo_1/utils.py)
- [`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)
- [`setup_brax_ubuntu.sh`](/D:/my_workspace/demo_1/setup_brax_ubuntu.sh)
- [`run_brax_paper_ubuntu.sh`](/D:/my_workspace/demo_1/run_brax_paper_ubuntu.sh)
- [`requirements.txt`](/D:/my_workspace/demo_1/requirements.txt)
- [`requirements-brax.txt`](/D:/my_workspace/demo_1/requirements-brax.txt)
- [`README.md`](/D:/my_workspace/demo_1/README.md)
- [`README_BRAX.md`](/D:/my_workspace/demo_1/README_BRAX.md)
- [`INTERFACE_MAP.md`](/D:/my_workspace/demo_1/INTERFACE_MAP.md)
- [`SERVER_DEPLOYMENT_LAYOUT.md`](/D:/my_workspace/demo_1/SERVER_DEPLOYMENT_LAYOUT.md)

## 4. 不建议上传到仓库或长期保留的目录

下面这些内容建议保留为服务器本地生成内容，而不是提交回 Git：

- `runtime/`
- `.venv-cec/`
- `.venv-brax/`
- `save_dir/`
- `tmp/`
- `pip-cache/`
- `__pycache__/`
- `logs/`

这些目录通常属于：

- 本地环境
- 中间缓存
- 实验输出
- 调度日志

## 5. CEC 任务的推荐部署方式

### 源码目录

建议直接在仓库根目录运行：

```bash
cd ~/projects/istratde-interface
```

### 环境目录

建议单独建一个环境：

```bash
python3.12 -m venv .venv-cec
source .venv-cec/bin/activate
```

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

### 推荐输出目录

当前脚本默认输出到仓库内的 `save_dir/`。
如果后续服务器要做大规模实验，建议把输出重定向到更独立的目录，例如：

```bash
mkdir -p ~/experiments/istratde-interface/outputs/cec
```

如果后续要进一步工程化，可以把 `save_dir` 抽成环境变量。

### 运行命令

完整跑 `F1 -> F15`：

```bash
source .venv-cec/bin/activate
python test.py
```

只跑部分函数：

```bash
source .venv-cec/bin/activate
FUN_ID_START=1 FUN_ID_END=3 python test.py
```

## 6. Brax 任务的推荐部署方式

### 环境目录

建议单独建立 Brax 环境：

```bash
python3.12 -m venv .venv-brax
source .venv-brax/bin/activate
```

或者直接使用仓库内脚本：

```bash
bash ./setup_brax_ubuntu.sh
```

### 运行命令

快速验证：

```bash
BRAX_ENVS=swimmer BRAX_TIME_BUDGET_MINUTES=5 bash ./run_brax_paper_ubuntu.sh
```

完整论文型运行：

```bash
bash ./run_brax_paper_ubuntu.sh
```

### 推荐输出目录

当前脚本默认输出到：

```text
save_dir/brax/ISTRATDE_<timestamp>/
```

建议服务器长期运行时额外保留：

- `logs/brax/`
- `outputs/brax/`

并把终端输出通过 `tee` 或 `nohup` 保存下来。

## 7. 适合服务器调度的运行方式

如果实验室服务器使用 `nohup`：

```bash
mkdir -p logs
nohup bash ./run_brax_paper_ubuntu.sh > logs/brax_run.log 2>&1 &
```

如果实验室服务器使用 `tmux`：

```bash
tmux new -s brax
bash ./run_brax_paper_ubuntu.sh
```

如果实验室服务器使用 `slurm`，建议后续再单独补一份 `sbatch` 脚本。

## 8. 推荐的最小可运行上传包

如果你的目标是“上传一个最小但能直接在服务器部署的版本”，建议至少包含：

```text
MMES/
cec2013lsgo/
istratde-main/src/istratde/
test.py
utils.py
brax_paper_benchmark.py
setup_brax_ubuntu.sh
run_brax_paper_ubuntu.sh
requirements.txt
requirements-brax.txt
README.md
README_BRAX.md
INTERFACE_MAP.md
SERVER_DEPLOYMENT_LAYOUT.md
```

## 9. 当前最推荐的接入方式

如果后续要真正接入实验室服务器，我建议按下面顺序推进：

1. 先部署 `CEC` 路径，验证 `iStratDE` 适配层是否稳定  
重点文件：[`MMES/istratde_optimizer.py`](/D:/my_workspace/demo_1/MMES/istratde_optimizer.py)、[`test.py`](/D:/my_workspace/demo_1/test.py)

2. 再部署 `Brax` 路径，使用 Ubuntu 独立环境运行  
重点文件：[`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)、[`setup_brax_ubuntu.sh`](/D:/my_workspace/demo_1/setup_brax_ubuntu.sh)

3. 最后再考虑服务器批量调度与日志系统整合  
例如：
- `nohup`
- `tmux`
- `slurm`

## 10. 当前边界提醒

### CEC

- 算法更新：GPU
- benchmark 评估：CPU

### Brax

- 已有独立论文复现实验入口
- 更适合 Ubuntu GPU 服务器
- 更推荐 Python 3.10 到 3.12

这份目录说明的核心目的，是让后续服务器接入时不需要再重新理解整个仓库，而是直接按模块拆分部署。
