import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ISTRATDE_SRC = ROOT / "istratde-main" / "src"
if str(ISTRATDE_SRC) not in sys.path:
    sys.path.insert(0, str(ISTRATDE_SRC))


def parse_env_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_hidden_dims(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class BraxRunConfig:
    env_names: list[str]
    time_budget_minutes: float
    pop_size: int
    max_episode_length: int
    num_episodes: int
    hidden_dims: list[int]
    log_interval_seconds: float
    seed: int
    save_html: bool
    backend: str | None


def load_config() -> BraxRunConfig:
    backend = os.environ.get("BRAX_BACKEND", "").strip() or None
    return BraxRunConfig(
        env_names=parse_env_list(os.environ.get("BRAX_ENVS", "swimmer,hopper,reacher")),
        time_budget_minutes=float(os.environ.get("BRAX_TIME_BUDGET_MINUTES", "60")),
        pop_size=int(os.environ.get("BRAX_POP_SIZE", "10000")),
        max_episode_length=int(os.environ.get("BRAX_MAX_EPISODE_LENGTH", "500")),
        num_episodes=int(os.environ.get("BRAX_NUM_EPISODES", "1")),
        hidden_dims=parse_hidden_dims(os.environ.get("BRAX_HIDDEN_DIMS", "32,32")),
        log_interval_seconds=float(os.environ.get("BRAX_LOG_INTERVAL_SECONDS", "60")),
        seed=int(os.environ.get("BRAX_SEED", "42")),
        save_html=parse_bool(os.environ.get("BRAX_SAVE_HTML", "0")),
        backend=backend,
    )


def ensure_dependencies() -> None:
    missing = []
    for module_name in ("torch", "jax", "jaxlib", "brax", "evox", "numpy", "matplotlib"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if not missing:
        return

    python_version = sys.version.split()[0]
    missing_str = ", ".join(missing)
    raise SystemExit(
        "Missing Brax dependencies: "
        f"{missing_str}. Current Python is {python_version}. "
        "This benchmark needs torch + evox + jax + jaxlib + brax + numpy + matplotlib. "
        "Note: Brax/JAX GPU support is typically easiest on Linux/WSL; "
        "current Windows Python 3.13 environments may require version adjustments."
    )


ensure_dependencies()

import numpy as np
import jax
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from brax import envs
from istratde.algorithms.pytorch.istratde import IStratDE
from istratde.problems.torch.brax import BraxProblem
from istratde.util import ParamsAndVector
from istratde.util.workflows import EvalMonitor, StdWorkflow


torch.set_float32_matmul_precision("high")


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float32))


def get_env_dimensions(env_name: str, backend: str | None) -> tuple[int, int]:
    env = envs.get_environment(env_name=env_name) if backend is None else envs.get_environment(
        env_name=env_name,
        backend=backend,
    )
    obs_dim = getattr(env, "observation_size", None)
    act_dim = getattr(env, "action_size", None)

    if callable(obs_dim):
        obs_dim = obs_dim()
    if callable(act_dim):
        act_dim = act_dim()

    if obs_dim is None:
        state = env.reset(jax.random.PRNGKey(0))
        obs_dim = int(np.prod(state.obs.shape))
    if act_dim is None:
        raise ValueError(f"Unable to resolve action dimension for Brax env `{env_name}`")

    return int(obs_dim), int(act_dim)


def make_output_root() -> Path:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_root = ROOT / "save_dir" / "brax" / f"ISTRATDE_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def write_reward_history(history: list[dict[str, float]], output_dir: Path) -> None:
    csv_path = output_dir / "reward_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "elapsed_seconds", "elapsed_minutes", "evaluations", "best_reward"],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_reward_history(history: list[dict[str, float]], output_dir: Path, env_name: str) -> None:
    elapsed_minutes = [row["elapsed_minutes"] for row in history]
    best_rewards = [row["best_reward"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(elapsed_minutes, best_rewards, linewidth=2.0)
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Best Reward")
    plt.title(f"iStratDE on Brax {env_name.capitalize()}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "reward_curve.pdf", bbox_inches="tight")
    plt.close()


def print_runtime_info(config: BraxRunConfig, device: str) -> None:
    print("Brax paper benchmark: ISTRATDE")
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch version: {torch.__version__}")
    print(f"jax version: {jax.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"active device: {torch.cuda.current_device()}")
        print(f"device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"execution device: {device}")
    print(f"envs: {', '.join(config.env_names)}")
    print(f"time budget per env: {config.time_budget_minutes:.2f} minutes")
    print(f"population size: {config.pop_size}")
    print(f"episode length: {config.max_episode_length}")
    print(f"num episodes: {config.num_episodes}")
    print(f"hidden dims: {config.hidden_dims}")
    print(f"brax backend: {config.backend or 'default'}")


def maybe_save_visualization(
    problem: BraxProblem,
    adapter: ParamsAndVector,
    best_vector: torch.Tensor,
    output_dir: Path,
    save_html: bool,
) -> None:
    if not save_html:
        return

    best_params = adapter.to_params(best_vector)
    html_content = problem.visualize(best_params, output_type="HTML")
    (output_dir / "best_policy.html").write_text(html_content, encoding="utf-8")


def run_single_env(
    env_name: str,
    config: BraxRunConfig,
    output_root: Path,
    device: str,
) -> dict[str, float | int | str]:
    env_output_dir = output_root / env_name
    env_output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim, act_dim = get_env_dimensions(env_name, config.backend)
    print("-" * 96)
    print(
        f"[Brax] env={env_name} obs_dim={obs_dim} act_dim={act_dim} "
        f"pop_size={config.pop_size} budget={config.time_budget_minutes:.2f}min"
    )

    torch.manual_seed(config.seed)
    model = PolicyNet(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=config.hidden_dims).to(device)
    adapter = ParamsAndVector(dummy_model=model)
    center = adapter.to_vector(dict(model.named_parameters()))
    lb = torch.full_like(center, -10.0)
    ub = torch.full_like(center, 10.0)

    algorithm = IStratDE(
        pop_size=config.pop_size,
        lb=lb,
        ub=ub,
    )
    problem = BraxProblem(
        policy=model,
        env_name=env_name,
        max_episode_length=config.max_episode_length,
        num_episodes=config.num_episodes,
        pop_size=config.pop_size,
        seed=config.seed,
        backend=config.backend,
        device=torch.device(device),
    )
    monitor = EvalMonitor(full_sol_history=False, full_fit_history=True)
    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
        solution_transform=adapter,
        device=device,
        opt_direction="max",
    )

    history: list[dict[str, float]] = []
    generation = 0
    evaluations = 0
    start_time = time.perf_counter()
    deadline = start_time + config.time_budget_minutes * 60.0
    next_log_time = start_time

    while True:
        now = time.perf_counter()
        if generation > 0 and now >= deadline:
            break

        if generation == 0:
            workflow.init_step()
        else:
            workflow.step()

        generation += 1
        evaluations += config.pop_size
        elapsed_seconds = time.perf_counter() - start_time
        best_reward = float(monitor.get_best_fitness().item())
        history.append(
            {
                "generation": generation,
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_seconds / 60.0,
                "evaluations": evaluations,
                "best_reward": best_reward,
            }
        )

        if elapsed_seconds >= (next_log_time - start_time) or generation == 1:
            print(
                f"[Brax][{env_name}] generation={generation} evaluations={evaluations} "
                f"elapsed={elapsed_seconds / 60.0:.2f}min best_reward={best_reward:.6f}"
            )
            next_log_time = start_time + elapsed_seconds + config.log_interval_seconds

    total_runtime = time.perf_counter() - start_time
    best_reward = float(monitor.get_best_fitness().item())
    best_vector = monitor.get_best_solution().detach().cpu()

    write_reward_history(history, env_output_dir)
    plot_reward_history(history, env_output_dir, env_name)
    torch.save(best_vector, env_output_dir / "best_solution.pt")
    maybe_save_visualization(problem, adapter, best_vector.to(device), env_output_dir, config.save_html)

    summary = {
        "env_name": env_name,
        "best_reward": best_reward,
        "generations": generation,
        "evaluations": evaluations,
        "runtime_seconds": total_runtime,
        "runtime_minutes": total_runtime / 60.0,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "pop_size": config.pop_size,
        "max_episode_length": config.max_episode_length,
        "num_episodes": config.num_episodes,
        "device": device,
    }
    (env_output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[Brax][{env_name}] completed runtime={total_runtime / 60.0:.2f}min "
        f"generations={generation} best_reward={best_reward:.6f}"
    )
    return summary


def main() -> None:
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    output_root = make_output_root()
    print_runtime_info(config, device)

    all_summaries = []
    for env_name in config.env_names:
        all_summaries.append(run_single_env(env_name, config, output_root, device))

    aggregate = {
        "config": asdict(config),
        "device": device,
        "output_root": str(output_root),
        "results": all_summaries,
    }
    (output_root / "summary_all.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("=" * 96)
    print(f"Brax benchmark finished. Results saved to: {output_root}")
    for summary in all_summaries:
        print(
            f"  {summary['env_name']}: best_reward={summary['best_reward']:.6f}, "
            f"runtime={summary['runtime_minutes']:.2f}min, generations={summary['generations']}"
        )


if __name__ == "__main__":
    main()
