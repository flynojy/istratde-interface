from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME_PYTHON = ROOT / "runtime" / "python" / "python.exe"
REQUIRED_PATHS = [
    ROOT / "runtime" / "python",
    ROOT / "istratde-main",
    ROOT / "cec2013lsgo",
    ROOT / "MMES",
    ROOT / "test.py",
    ROOT / "run_test_istratde.bat",
]


def print_status(ok: bool, label: str, detail: str = ""):
    status = "OK" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def main():
    print("Portable Environment Check")
    print(f"root: {ROOT}")
    print(f"python: {sys.executable}")
    print("-" * 72)

    all_ok = True

    for path in REQUIRED_PATHS:
        exists = path.exists()
        print_status(exists, f"path exists: {path.relative_to(ROOT)}")
        all_ok &= exists

    using_portable_python = Path(sys.executable).resolve() == RUNTIME_PYTHON.resolve()
    print_status(using_portable_python, "using bundled runtime", str(RUNTIME_PYTHON))
    all_ok &= using_portable_python

    print("-" * 72)

    try:
        import torch

        print_status(True, "import torch", torch.__version__)
        cuda_available = torch.cuda.is_available()
        print_status(cuda_available, "cuda available")
        if cuda_available:
            print_status(True, "cuda device count", str(torch.cuda.device_count()))
            print_status(True, "cuda device name", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("INFO: GPU unavailable. The project can still run on CPU, but slower.")
    except Exception as exc:
        print_status(False, "import torch", f"{type(exc).__name__}: {exc}")
        all_ok = False

    for module_name in ["evox", "numba", "numpy", "scipy", "h5py", "matplotlib", "tensorboard"]:
        try:
            __import__(module_name)
            print_status(True, f"import {module_name}")
        except Exception as exc:
            print_status(False, f"import {module_name}", f"{type(exc).__name__}: {exc}")
            all_ok = False

    print("-" * 72)

    output_dir = ROOT / "save_dir" / "baseline"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / "__write_test.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print_status(True, "output directory writable", str(output_dir))
    except Exception as exc:
        print_status(False, "output directory writable", f"{type(exc).__name__}: {exc}")
        all_ok = False

    mpl_dir = ROOT / "runtime" / "matplotlib"
    pycache_dir = ROOT / "runtime" / "pycache"
    tmp_dir = ROOT / "runtime" / "tmp"
    for path in [mpl_dir, pycache_dir, tmp_dir]:
        try:
            path.mkdir(parents=True, exist_ok=True)
            print_status(True, "runtime dir writable", str(path.relative_to(ROOT)))
        except Exception as exc:
            print_status(False, "runtime dir writable", f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
            all_ok = False

    print("-" * 72)
    if all_ok:
        print("Environment check passed.")
        return 0

    print("Environment check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
