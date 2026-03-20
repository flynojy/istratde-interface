import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "save_dir" / "baseline"


def find_latest_result(algorithm: str, fun_id: int):
    pattern = f"{algorithm}_*"
    candidates = []
    for algo_dir in BASELINE_DIR.glob(pattern):
        if not algo_dir.is_dir():
            continue
        for result_path in algo_dir.glob(f"F{fun_id}_*/result_record.txt"):
            candidates.append(result_path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_result_record(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    final_match = re.search(r"Final\(([^|]+)\|([0-9,]+)\)\s+([0-9eE+\-.]+)", text)
    time_match = re.search(r"Avg Time\(s\)\s+([0-9eE+\-.]+)", text)
    if final_match is None:
        raise ValueError(f"Could not parse final result from {path}")

    return {
        "scale": final_match.group(1),
        "evaluations": final_match.group(2),
        "final_value": float(final_match.group(3)),
        "avg_time": float(time_match.group(1)) if time_match else None,
        "path": str(path),
    }


def main():
    fun_id = 15
    algorithms = ["MMES", "ISTRATDE"]
    rows = []

    for algorithm in algorithms:
        path = find_latest_result(algorithm, fun_id)
        if path is None:
            print(f"{algorithm}: no result found for F{fun_id}")
            continue
        result = parse_result_record(path)
        rows.append((algorithm, result))

    if not rows:
        print("No comparison data found.")
        return

    print(f"Latest comparison for F{fun_id}")
    print("-" * 96)
    print(f"{'Algorithm':<12}{'Final Value':<20}{'Avg Time(s)':<16}{'Evaluations':<16}Path")
    print("-" * 96)
    for algorithm, result in rows:
        avg_time = "N/A" if result["avg_time"] is None else f"{result['avg_time']:.6f}"
        print(
            f"{algorithm:<12}"
            f"{result['final_value']:<20.6e}"
            f"{avg_time:<16}"
            f"{result['evaluations']:<16}"
            f"{result['path']}"
        )

    if len(rows) == 2:
        winner = min(rows, key=lambda item: item[1]["final_value"])
        print("-" * 96)
        print(f"Lower final objective value wins: {winner[0]} ({winner[1]['final_value']:.6e})")


if __name__ == "__main__":
    main()
