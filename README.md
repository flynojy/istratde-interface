# iStratDE Interface

这是一个面向实验复现和工程接入的适配仓库，目标是把 `iStratDE` 接入到本地已有的 `problem/options -> optimize()` 优化接口中，同时保留 `MMES` 作为基线算法，方便在同一套测试框架下直接对比。

本仓库目前主要围绕 `CEC2013 LSGO` 测试问题展开，并支持：

- 使用 `ISTRATDE` 作为优化器运行现有测试脚本
- 使用 `MMES` 作为基线进行公平对照
- 输出结果表、收敛曲线和简单 profiling 信息
- 在 Windows 环境下通过 PyTorch CUDA 后端调用 GPU

## 项目亮点

- 保留原有接口习惯  
  外层仍然使用 `problem/options -> optimize()` 风格，不需要把整套调用链改写成原始 iStratDE 示例形式。

- 支持 iStratDE / MMES 直接切换  
  同一个 `test.py` 入口下即可切换算法，方便做对照实验。

- 支持 GPU 算法后端  
  `iStratDE` 当前通过 PyTorch CUDA 后端运行，能够在 NVIDIA GPU 上完成种群更新与演化过程。

- 自带实验脚本  
  提供单算法运行脚本和对照运行脚本，方便快速复现实验。

## 当前实现边界

当前这套接法是“混合模式”：

- `iStratDE` 算法更新：GPU
- `cec2013lsgo` 目标函数评估：CPU

也就是说，当前已经启用了 CUDA，但 GPU 利用率不一定很高。这是因为 `cec2013lsgo` 这部分仍然主要基于 `NumPy + Numba`，瓶颈通常在 benchmark 评估而不是优化器本身。

如果后续希望进一步放大 iStratDE 的 GPU 优势，需要继续把 benchmark 评估层改写成 `torch` 或 `jax` 原生实现。

## 仓库结构

```text
demo_1/
├─ MMES/                    # 本地优化器框架与 iStratDE 适配层
├─ cec2013lsgo/            # CEC2013 LSGO 测试函数
├─ istratde-main/          # 上游 iStratDE 源码
├─ test.py                 # 主实验入口
├─ utils.py                # 结果记录与绘图工具
├─ run_test_istratde.bat   # 单独运行 iStratDE
├─ run_compare_f15.bat     # MMES vs ISTRATDE 对照入口
├─ compare_results.py      # 汇总最近一次对照结果
├─ check_portable_env.py   # 环境检查脚本
├─ requirements.txt        # 除 PyTorch 外的依赖
└─ README.md
```

## 环境要求

推荐环境：

- Windows 10 / 11
- Python 3.13
- NVIDIA GPU
- 正常安装的 NVIDIA 驱动

## 安装方式

建议使用独立 Python 环境。

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

可通过环境变量修改实验配置：

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

切换回 MMES：

```powershell
$env:DEMO_OPTIMIZER="MMES"
.\run_test_istratde.bat
```

## 运行输出说明

脚本启动后会输出：

- 当前优化器
- iStratDE 后端类型
- PyTorch 版本
- CUDA 是否可用
- 当前 GPU 名称

实验过程中还会输出：

- `generation`
- `evaluations`
- `best_so_far_y`

运行结束后会输出一段 profiling 信息，例如：

- 总 step 时间
- evaluation pipeline 时间
- algorithm/framework 时间

这些信息有助于判断时间主要花在：

- 优化器本身
- 还是 benchmark 评估

## 结果文件说明

运行结束后，通常会在 `save_dir/baseline/...` 下生成：

- `result_record.txt`
- `evaluation_curves_best_so_far.pdf`
- `evaluation_curves_best_so_far.png`

这些结果表示的是优化问题中的：

- 函数评估次数（FEs）
- best-so-far objective value

它不是分类准确率或模型“打分”，而是 benchmark 的目标函数值。  
在当前 `CEC2013LSGO` 问题里，一般是 **越小越好**。

## 主要实现文件

- `MMES/istratde_optimizer.py`  
  将 iStratDE 适配到本地 `Optimizer` 风格接口

- `test.py`  
  主实验入口，支持 `MMES` / `ISTRATDE` 切换

- `utils.py`  
  结果记录、绘图和汇总工具

- `compare_results.py`  
  汇总最近一次 MMES 与 ISTRATDE 的对照结果

## 建议忽略的本地产物

以下目录或文件通常不建议上传：

- `runtime/`
- `save_dir/`
- `tmp/`
- `pip-cache/`
- `.venv-istratde/`
- `__pycache__/`

这些内容要么体积大，要么是本地缓存或实验产物，不适合作为源码仓库的一部分。

## 后续可扩展方向

- 将 `cec2013lsgo` 评估层改写为 GPU 原生实现
- 增加更多函数的自动对照实验
- 增加多次独立运行统计和统一汇总表
- 增加更完整的实验配置管理

## 致谢

- 上游 `iStratDE`
- `MMES`
- `CEC2013 LSGO` benchmark

本仓库的工作重点是接口适配、工程集成与实验组织，而不是替代这些原始项目本身。
