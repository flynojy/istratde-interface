# Portable Runtime Notes

This project now includes a bundled Python runtime under `runtime/python`.

## Recommended entrypoint

Run:

```bat
run_test_istratde.bat
```

Before first use on another machine, run:

```bat
check_portable_env.bat
```

The launcher keeps runtime artifacts inside `demo_1/runtime`:

- temp files
- pip cache
- matplotlib cache
- Python bytecode cache

## Default behavior

- Optimizer: `ISTRATDE`
- Backend: `torch`
- Process pool: disabled by default for portability

## Common overrides

Examples:

```bat
set POP_SIZE=512
set MAX_FES=2E5
set CYCLE_NUM=3
set USE_PROCESS_POOL=1
run_test_istratde.bat
```

To switch back to MMES:

```bat
set DEMO_OPTIMIZER=MMES
run_test_istratde.bat
```

## Delivery checklist

For another Windows machine:

1. Copy the whole `demo_1` folder.
2. Ensure an NVIDIA driver is installed if GPU execution is desired.
3. Run `check_portable_env.bat`.
4. Run `run_test_istratde.bat`.

## Fair comparison

Run MMES and ISTRATDE sequentially under the same settings:

```bat
run_compare_f15.bat
```

Optional overrides:

```bat
set POP_SIZE=1000
set MAX_FES=1E6
set CYCLE_NUM=1
set VERBOSE_EVERY=100
run_compare_f15.bat
```
