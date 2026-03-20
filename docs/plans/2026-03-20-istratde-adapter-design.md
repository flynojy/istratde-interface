# iStratDE Adapter Design

## Goal

Preserve the local `problem/options -> optimize()` contract while allowing iStratDE to run under the same experiment harness that currently uses MMES.

## Decision

Use an adapter class instead of rewriting the existing experiment scripts or benchmark wrappers.

- New class: `MMES/istratde_optimizer.py::IStratDEOptimizer`
- Base class: `MMES.optimizer.Optimizer`
- Input contract: unchanged `problem` and `options` dictionaries
- Output contract: unchanged MMES-style results dictionary

## Architecture

The adapter reuses `Optimizer` for:

- fitness evaluation accounting
- best-so-far tracking
- termination checks
- compressed fitness history output

It delegates search to iStratDE through one of two backends:

- `backend="jax"`:
  uses `istratde.util.standard.StdWorkflow`
- `backend="torch"`:
  uses `istratde.util.workflows.StdWorkflow`

The bridge layer is a small external-problem wrapper that forwards batched candidate solutions into the existing `fitness_function`.

## Mapping Rules

- `n_individuals` or `pop_size` -> iStratDE population size
- `max_function_evaluations` -> stop condition driven by local `Optimizer`
- `lower_boundary` / `upper_boundary` -> `lb` / `ub`
- `mean` + `sigma`:
  - JAX: warm-start is injected by overriding the initialized population state
  - Torch: passed through to the algorithm constructor

## Known Constraints

- JAX backend requires `jax` and `jaxlib`
- Torch backend requires `torch` and `evox`
- The upstream repository mixes JAX and Torch package imports, so the adapter uses lazy imports and direct file loading to reduce cross-backend coupling
- The adapter keeps the outer contract stable, but the internal evaluation lifecycle still follows iStratDE semantics
