import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.data import get_recommended_days_for_granularity
from utils.granularity import Granularity

MODELS = {
    "xgb": "train_xgb.py",
    "rf": "train_rf.py",
    "linear": "train_baseline_linear.py",
    "ebm": "train_ebm.py",
}

GRANULARITIES = ["H", "D", "W", "M", "Y"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train all models at all granularities")
    parser.add_argument("--granularities", "-g", nargs="+", default=GRANULARITIES, choices=GRANULARITIES)
    parser.add_argument("--models", "-m", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", "-s", type=str, default="real", choices=["real", "synthetic"])
    return parser.parse_args()


def train_model(model, granularity, days, seed, source="real"):
    script = MODELS[model]
    script_path = Path(__file__).parent / script
    python = sys.executable

    cmd = [
        python, str(script_path),
        "--granularity", granularity,
        "--days", str(days),
        "--seed", str(seed),
        "--source", source,
    ]

    result = {
        "model": model,
        "granularity": granularity,
        "success": False,
        "output": "",
        "error": "",
    }

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        result["output"] = proc.stdout
        result["error"] = proc.stderr
        result["success"] = proc.returncode == 0
    except subprocess.TimeoutExpired:
        result["error"] = "Training timed out after 10 minutes"
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    args = parse_args()

    print("=" * 60)
    print("Training all models")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Granularities: {args.granularities}")
    print(f"Source: {args.source}")
    print(f"Parallel: {args.parallel}")
    print("=" * 60)
    print()

    tasks = []
    for granularity in args.granularities:
        days = get_recommended_days_for_granularity(Granularity.from_code(granularity))
        for model in args.models:
            tasks.append((model, granularity, days, args.seed, args.source))

    total = len(tasks)
    results = []

    if args.parallel:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(train_model, *task): task
                for task in tasks
            }
            for i, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                result = future.result()
                results.append(result)
                status = "OK" if result["success"] else "FAILED"
                print(f"[{i}/{total}] [{status}] {result['model']} @ {result['granularity']}")
    else:
        for i, (model, granularity, days, seed, source) in enumerate(tasks, 1):
            print(f"\n[{i}/{total}] {'=' * 40}")
            print(f"Training {model} @ {granularity} ({days} days, source={source})")
            print("=" * 40)

            result = train_model(model, granularity, days, seed, source)
            results.append(result)

            if result["success"]:
                lines = result["output"].strip().split("\n")
                for line in lines[-10:]:
                    print(line)
            else:
                print(f"FAILED: {result['error']}")

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Successful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  - {r['model']} @ {r['granularity']}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  - {r['model']} @ {r['granularity']}: {r['error'][:50]}")


if __name__ == "__main__":
    main()
