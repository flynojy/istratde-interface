# iStratDE Interface Adapter

这是一个面向实验复现与服务器部署的 iStratDE 适配仓库。

本仓库的核心目标不是重写上游 `istratde`，而是把它整理成实验室现有流程更容易接入的形式：

- 保留本地已有的 `problem/options -> optimize()` 调用习惯
- 在 `CEC2013 LSGO` 上用统一入口对接 `MMES` 和 `iStratDE`
- 新增独立的 `Brax` 论文复现实验入口
- 为后续迁移到 Ubuntu / 实验室 GPU 服务器做准备

## 当前完成的能力

- `iStratDE` 已通过适配器接入本地优化器接口
- `CEC2013LSGO` 已支持批量跑 `F1 -> F15`
- `iStratDE` 已支持 PyTorch CUDA 后端
- 已支持基础 profiling 输出，区分算法时间和评估时间
- 已补齐 Brax 论文复现实验入口
- 已补齐 Ubuntu 环境脚本，便于后续部署到服务器

## 核心接口总览

### 1. 统一优化器接口

核心适配文件：

- [`MMES/istratde_optimizer.py`](/D:/my_workspace/demo_1/MMES/istratde_optimizer.py)

外部接口保持为：

```python
from MMES.istratde_optimizer import IStratDEOptimizer

optimizer = IStratDEOptimizer(problem, options)
results = optimizer.optimize()
```

其中：

- `problem` 负责提供目标函数、维度和边界
- `options` 负责提供种群规模、随机种子、后端和终止条件
- `results` 返回 best fitness、评估次数、运行时间和 profiling 信息

这层是本仓库最重要的接口整理结果，也是后续接入实验室服务器时最值得保留的部分。

### 2. CEC 实验入口

统一实验入口：

- [`test.py`](/D:/my_workspace/demo_1/test.py)

相关脚本：

- [`run_test_istratde.bat`](/D:/my_workspace/demo_1/run_test_istratde.bat)
- [`run_compare_f15.bat`](/D:/my_workspace/demo_1/run_compare_f15.bat)
- [`compare_results.py`](/D:/my_workspace/demo_1/compare_results.py)

当前默认支持：

- `F1 -> F15`
- `MMES / ISTRATDE` 切换
- 进度输出
- profiling 输出
- 结果曲线和文本统计

### 3. Brax 论文复现实验入口

核心文件：

- [`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)

运行脚本：

- Windows: [`run_brax_paper.bat`](/D:/my_workspace/demo_1/run_brax_paper.bat)
- Ubuntu: [`setup_brax_ubuntu.sh`](/D:/my_workspace/demo_1/setup_brax_ubuntu.sh)
- Ubuntu: [`run_brax_paper_ubuntu.sh`](/D:/my_workspace/demo_1/run_brax_paper_ubuntu.sh)

当前设计目标是贴近论文中的 Brax 实验：

- 环境：`swimmer`, `hopper`, `reacher`
- 停止条件：按运行时长，而不是固定函数评估次数
- 输出：reward history、曲线图、summary、最佳策略权重

## 仓库结构

```text
demo_1/
|-- MMES/
|   |-- optimizer.py
|   |-- mmes.py
|   `-- istratde_optimizer.py
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
`-- INTERFACE_MAP.md
```

## 快速开始

### Windows 下运行 CEC

安装 CUDA 版 PyTorch：

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装其余依赖：

```powershell
python -m pip install -r requirements.txt
```

直接运行：

```powershell
.\run_test_istratde.bat
```

### Ubuntu 下运行 Brax

推荐环境：

- Ubuntu 22.04 / 24.04
- Python 3.10 到 3.12
- NVIDIA 驱动正常
- CUDA 可用

安装环境：

```bash
bash ./setup_brax_ubuntu.sh
```

快速验证：

```bash
BRAX_ENVS=swimmer BRAX_TIME_BUDGET_MINUTES=5 bash ./run_brax_paper_ubuntu.sh
```

完整运行：

```bash
bash ./run_brax_paper_ubuntu.sh
```

## 服务器接入建议

如果后续要接入实验室服务器，建议直接围绕下面这几层展开：

1. 保留本仓库的适配层  
重点保留 [`MMES/istratde_optimizer.py`](/D:/my_workspace/demo_1/MMES/istratde_optimizer.py)、[`test.py`](/D:/my_workspace/demo_1/test.py)、[`brax_paper_benchmark.py`](/D:/my_workspace/demo_1/brax_paper_benchmark.py)。

2. 优先迁移到 Ubuntu 环境  
尤其是 Brax。`Windows + Python 3.13` 不适合做 Brax/JAX 论文复现，服务器推荐 `Ubuntu + Python 3.10~3.12 + CUDA`。

3. 分开维护两类任务  
- `CEC2013LSGO`：保留当前本地接口风格
- `Brax`：保留当前独立实验入口风格

4. 后续可继续做的优化  
- 将 `cec2013lsgo` 评估层改写为 GPU 原生实现
- 补齐服务器批量调度脚本
- 接入实验日志系统

## 当前边界

### CEC

当前链路是：

- `iStratDE` 算法侧：GPU
- `cec2013lsgo` 评估侧：CPU

所以现在已经能跑 GPU，但不是全链路纯 GPU。

### Brax

当前仓库已经补齐 Brax 入口，但真实运行更推荐在 Ubuntu 上完成。

## 相关文档

- Brax 说明：[`README_BRAX.md`](/D:/my_workspace/demo_1/README_BRAX.md)
- 接口映射：[`INTERFACE_MAP.md`](/D:/my_workspace/demo_1/INTERFACE_MAP.md)

## 致谢

- 上游 [`istratde-main`](/D:/my_workspace/demo_1/istratde-main)
- `MMES`
- `CEC2013 LSGO`
- `Brax`
