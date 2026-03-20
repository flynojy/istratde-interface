# demo_1

这是一个将 `iStratDE` 适配到本地 `problem/options -> optimize()` 接口体系中的实验仓库，同时保留 `MMES` 作为基线算法，方便在同一套 CEC2013 LSGO 测试框架下做对比。

## 项目目标

本仓库主要完成了以下工作：

- 保留原有 `MMES` 风格调用接口
- 新增 `iStratDE` 适配器，使其可以直接接入现有测试脚本
- 支持 `MMES` 与 `ISTRATDE` 在相同预算下做对照实验
- 支持输出运行曲线、结果记录和简单 profiling 信息

## 目录说明

- `MMES/`
  本地优化器框架以及 `IStratDEOptimizer` 适配器
- `cec2013lsgo/`
  CEC2013 大规模全局优化测试函数实现
- `istratde-main/`
  上游 iStratDE 源码
- `test.py`
  统一实验入口
- `run_test_istratde.bat`
  单独运行 iStratDE 的启动脚本
- `run_compare_f15.bat`
  顺序运行 MMES 与 ISTRATDE 的对照脚本
- `compare_results.py`
  汇总最近一次 MMES 与 ISTRATDE 的结果

## 当前 GPU 使用方式说明

当前这套接法是“混合模式”：

- `iStratDE` 算法更新：走 PyTorch CUDA 后端，使用 GPU
- `cec2013lsgo` 目标函数评估：主要仍是 NumPy + Numba，在 CPU 上运行

所以你会看到：

- CUDA 确实启用
- GPU 可以参与算法部分
- 但 GPU 占用不一定很高

这是因为当前瓶颈通常在 benchmark 评估，而不是算法本体。

## 推荐环境

- Windows 10 / 11
- Python 3.13
- NVIDIA GPU
- 正常安装的 NVIDIA 驱动

## 环境准备

建议先创建一个新的 Python 环境，然后安装依赖。

### 1. 安装 CUDA 版 PyTorch

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 安装其余依赖

```powershell
python -m pip install -r requirements.txt
```

## 快速开始

### 运行 iStratDE

```powershell
.\run_test_istratde.bat
```

### 运行 MMES 与 ISTRATDE 对照实验

```powershell
.\run_compare_f15.bat
```

## 常用参数

可以通过环境变量控制实验配置。

```powershell
$env:DEMO_OPTIMIZER="ISTRATDE"
$env:ISTRATDE_BACKEND="torch"
$env:POP_SIZE="1000"
$env:MAX_FES="1E6"
$env:CYCLE_NUM="1"
$env:FUN_ID_START="15"
$env:FUN_ID_END="15"
$env:VERBOSE_EVERY="100"
.\run_test_istratde.bat
```

### 切换回 MMES

```powershell
$env:DEMO_OPTIMIZER="MMES"
.\run_test_istratde.bat
```

## 运行时输出说明

启动脚本会打印以下信息：

- 当前使用的优化器
- iStratDE 后端类型
- PyTorch 版本
- CUDA 是否可用
- 当前 GPU 名称

此外还会输出一段 profiling 信息，例如：

- 总 step 时间
- evaluation pipeline 时间
- algorithm/framework 时间

这能帮助判断时间主要花在：

- 算法本身
- 还是 benchmark 评估

## 结果文件说明

运行结束后，通常会在 `save_dir/baseline/...` 下生成：

- `result_record.txt`
- `evaluation_curves_best_so_far.pdf`
- `evaluation_curves_best_so_far.png`

其中图表和结果不是“模型准确率”，而是优化问题中的：

- 函数评估次数（FEs）
- best-so-far objective value

对于当前 CEC2013LSGO 问题，一般是 **越小越好**。

## 主要实现文件

- `MMES/istratde_optimizer.py`
  iStratDE 适配层
- `test.py`
  主实验入口
- `utils.py`
  结果记录与作图工具
- `compare_results.py`
  对比最近一次 MMES 与 ISTRATDE 的结果

## 上传 GitHub 时建议忽略

以下内容通常不建议上传：

- `runtime/`
- `save_dir/`
- `tmp/`
- `pip-cache/`
- `.venv-istratde/`
- `__pycache__/`

这些目录要么体积大，要么是本地运行产物，不适合作为源码仓库内容。

## 建议的 GitHub 上传内容

建议保留并上传：

- `MMES/`
- `cec2013lsgo/`
- `istratde-main/`
- `test.py`
- `utils.py`
- `run_test_istratde.bat`
- `run_compare_f15.bat`
- `compare_results.py`
- `check_portable_env.py`
- `requirements.txt`
- `.gitignore`
- `README.md`
