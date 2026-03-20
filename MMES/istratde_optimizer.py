import importlib
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from MMES.optimizer import Optimizer


class IStratDEOptimizer(Optimizer):
    """Adapter that keeps the local Optimizer interface while delegating search to iStratDE.

    The public contract intentionally matches the MMES-style usage pattern:

    ```python
    optimizer = IStratDEOptimizer(problem, options)
    results = optimizer.optimize()
    ```

    Supported options:
    - ``backend``: ``"jax"`` or ``"torch"``, defaults to ``"jax"``
    - ``n_individuals`` / ``pop_size``: population size used by iStratDE
    - ``seed_rng``: random seed
    - ``mean`` and ``sigma``: optional warm-start initialization
    - ``device``: torch device when ``backend="torch"``
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.backend = options.get("backend", "jax").lower()
        self.pop_size = options.get("pop_size", self.n_individuals)
        if self.pop_size is None:
            self.pop_size = 4 + int(3 * np.log(self.ndim_problem))
        if self.pop_size < 4:
            raise ValueError(f"pop_size must be >= 4, got {self.pop_size}")
        self.n_individuals = self.pop_size
        self.device = options.get("device")
        self._fitness_archive: list[float] = []
        self._istratde_src = Path(__file__).resolve().parent.parent / "istratde-main" / "src"
        self._profile = {}

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        self._fitness_archive = []
        self._profile = {
            "step_time": 0.0,
            "evaluation_time": 0.0,
            "evaluation_calls": 0,
            "generations": 0,
        }

        if self.backend == "jax":
            return self._optimize_jax(args=args)
        if self.backend == "torch":
            return self._optimize_torch(args=args)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _optimize_jax(self, args=None):
        jax, jnp, ProblemBase, StdWorkflow, StdSOMonitor, IStratDE = self._load_jax_backend()

        adapter = self

        class ExternalBatchProblem(ProblemBase):
            def __init__(self):
                pass

            def evaluate(self, state, pop):
                eval_start = time.perf_counter()
                pop_np = np.asarray(pop)
                fitness = adapter._evaluate_fitness(pop_np, args=args)
                adapter._fitness_archive.extend(np.asarray(fitness).reshape(-1).tolist())
                adapter._profile["evaluation_time"] += time.perf_counter() - eval_start
                adapter._profile["evaluation_calls"] += 1
                return jnp.asarray(fitness), state

        lb = jnp.asarray(self.lower_boundary)
        ub = jnp.asarray(self.upper_boundary)
        algorithm = IStratDE(lb=lb, ub=ub, pop_size=self.pop_size)
        monitor = StdSOMonitor(record_fit_history=False)
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=ExternalBatchProblem(),
            monitor=monitor,
        )

        seed = int(self.seed_rng if self.seed_rng is not None else 42)
        state = workflow.init(jax.random.PRNGKey(seed))
        state = self._override_jax_initial_state(state, jnp)
        generation = 0

        while not self.termination_signal:
            step_start = time.perf_counter()
            state = workflow.step(state)
            generation += 1
            self._profile["step_time"] += time.perf_counter() - step_start
            self._profile["generations"] = generation
            self._report_progress(generation)
            if self._check_terminations():
                break

        results = self._collect(self._fitness_archive)
        results["backend"] = "jax"
        results["raw_state"] = state
        results["raw_monitor"] = monitor
        return results

    def _optimize_torch(self, args=None):
        torch, EvoxProblem, EvalMonitor, StdWorkflow, IStratDE = self._load_torch_backend()

        adapter = self

        class ExternalBatchProblem(EvoxProblem):
            def __init__(self):
                super().__init__()

            def evaluate(self, pop):
                eval_start = time.perf_counter()
                pop_np = pop.detach().cpu().numpy()
                fitness = adapter._evaluate_fitness(pop_np, args=args)
                adapter._fitness_archive.extend(np.asarray(fitness).reshape(-1).tolist())
                adapter._profile["evaluation_time"] += time.perf_counter() - eval_start
                adapter._profile["evaluation_calls"] += 1
                return torch.as_tensor(fitness, device=pop.device, dtype=pop.dtype)

        seed = int(self.seed_rng if self.seed_rng is not None else 42)
        torch.manual_seed(seed)
        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        lb = torch.as_tensor(self.lower_boundary, dtype=torch.float32, device=device)
        ub = torch.as_tensor(self.upper_boundary, dtype=torch.float32, device=device)
        mean = self._normalized_mean()
        sigma = self.options.get("sigma")
        mean_tensor = None if mean is None else torch.as_tensor(mean, dtype=torch.float32, device=device)
        stdev_tensor = None
        if sigma is not None:
            stdev_tensor = torch.full((self.ndim_problem,), float(sigma), dtype=torch.float32, device=device)

        algorithm = IStratDE(
            pop_size=self.pop_size,
            lb=lb,
            ub=ub,
            mean=mean_tensor,
            stdev=stdev_tensor,
            device=device,
        )
        monitor = EvalMonitor(full_fit_history=False, full_sol_history=False, device=device)
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=ExternalBatchProblem(),
            monitor=monitor,
            device=device,
        )
        generation = 0

        while not self.termination_signal:
            self._sync_torch(torch)
            step_start = time.perf_counter()
            workflow.step()
            self._sync_torch(torch)
            generation += 1
            self._profile["step_time"] += time.perf_counter() - step_start
            self._profile["generations"] = generation
            self._report_progress(generation)
            if self._check_terminations():
                break

        results = self._collect(self._fitness_archive)
        results["backend"] = "torch"
        results["device"] = str(device)
        results["cuda_available"] = bool(torch.cuda.is_available())
        results["raw_monitor"] = monitor
        results["raw_state"] = None
        return results

    def _override_jax_initial_state(self, state, jnp):
        mean = self._normalized_mean()
        sigma = self.options.get("sigma")
        if mean is None or sigma is None:
            return state

        rng = np.random.default_rng(self.seed_initialization)
        population = rng.normal(loc=mean, scale=float(sigma), size=(self.pop_size, self.ndim_problem))
        population = np.clip(population, self.lower_boundary, self.upper_boundary)
        return state.update(
            population=jnp.asarray(population),
            fitness=jnp.full((self.pop_size,), jnp.inf),
            best_index=0,
            start_index=0,
        )

    def _normalized_mean(self):
        mean = self.options.get("mean")
        if mean is None:
            return None
        mean = np.asarray(mean, dtype=float)
        if mean.ndim == 2 and mean.shape[0] == 1:
            mean = mean[0]
        if mean.shape != (self.ndim_problem,):
            raise ValueError(
                f"mean must have shape ({self.ndim_problem},) or (1, {self.ndim_problem}), got {mean.shape}"
            )
        return mean

    def _load_jax_backend(self):
        self._ensure_istratde_path()
        try:
            jax = importlib.import_module("jax")
            jnp = importlib.import_module("jax.numpy")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "JAX backend requires `jax` and `jaxlib` to be installed."
            ) from exc

        try:
            util_module = importlib.import_module("istratde.util")
            standard_module = importlib.import_module("istratde.util.standard")
            monitor_module = importlib.import_module("istratde.util.std_so_monitor")
            algo_module = importlib.import_module("istratde.algorithms.jax.istratde")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import iStratDE JAX modules. Make sure `istratde-main/src` and its dependencies are available."
            ) from exc

        return (
            jax,
            jnp,
            util_module.Problem,
            standard_module.StdWorkflow,
            monitor_module.StdSOMonitor,
            algo_module.IStratDE,
        )

    def _load_torch_backend(self):
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Torch backend requires `torch` to be installed.") from exc

        try:
            evox_core = importlib.import_module("evox.core")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Torch backend requires `evox` to be installed.") from exc

        workflow_dir = self._istratde_src / "istratde" / "util" / "workflows"
        algo_file = self._istratde_src / "istratde" / "algorithms" / "pytorch" / "istratde.py"
        workflow_file = workflow_dir / "std_workflow.py"
        monitor_file = workflow_dir / "eval_monitor.py"

        algo_module = self._load_module_from_path("_istratde_torch_algorithm", algo_file)
        workflow_module = self._load_module_from_path("_istratde_torch_workflow", workflow_file)
        monitor_module = self._load_module_from_path("_istratde_torch_monitor", monitor_file)

        return (
            torch,
            evox_core.Problem,
            monitor_module.EvalMonitor,
            workflow_module.StdWorkflow,
            algo_module.IStratDE,
        )

    def _load_module_from_path(self, module_name: str, path: Path):
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from path: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_istratde_path(self):
        src = str(self._istratde_src)
        if src not in sys.path:
            sys.path.insert(0, src)

    def _collect(self, fitness):
        results = super()._collect(fitness)
        results["backend"] = self.backend
        results["pop_size"] = self.pop_size
        results["steps"] = self._estimated_steps()
        step_time = self._profile.get("step_time", 0.0)
        evaluation_time = self._profile.get("evaluation_time", 0.0)
        algorithm_time = max(step_time - evaluation_time, 0.0)
        if step_time > 0:
            evaluation_share = evaluation_time / step_time
            algorithm_share = algorithm_time / step_time
        else:
            evaluation_share = 0.0
            algorithm_share = 0.0
        results["profiling"] = {
            "step_time": step_time,
            "evaluation_time": evaluation_time,
            "algorithm_time": algorithm_time,
            "evaluation_share": evaluation_share,
            "algorithm_share": algorithm_share,
            "evaluation_calls": self._profile.get("evaluation_calls", 0),
            "generations": self._profile.get("generations", 0),
        }
        return results

    def _report_progress(self, generation: int):
        if not self.verbose:
            return

        should_print = (generation % int(self.verbose) == 0) or (self.n_function_evaluations >= self.max_function_evaluations)
        if not should_print:
            return

        best_value = float(self.best_so_far_y) if np.isfinite(self.best_so_far_y) else self.best_so_far_y
        print(
            f"[iStratDE] generation={generation} "
            f"evaluations={self.n_function_evaluations} "
            f"best_so_far_y={best_value:.6e}"
        )

    def _sync_torch(self, torch):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _estimated_steps(self):
        if not np.isfinite(self.max_function_evaluations):
            return None
        return int(math.ceil(float(self.max_function_evaluations) / float(self.pop_size)))
