"""
Train all models at all granularities.

Usage:
    python train_all.py                    # Train everything
    python train_all.py --granularities H D  # Only hourly and daily
    python train_all.py --models xgb rf    # Only XGBoost and RF
    python train_all.py --parallel         # Run in parallel (faster but more memory)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Available models and their training scripts
MODELS = {
    "xgb": "train_xgb.py",
    "rf": "train_rf.py",
    "linear": "train_baseline_linear.py",
}

# Available granularities
GRANULARITIES = ["H", "D", "W", "M", "Y"]

# Days of data per granularity (reasonable defaults for synthetic data)
DAYS_PER_GRANULARITY = {
    "H": 90,      # 3 months
    "D": 365,     # 1 year
    "W": 730,     # 2 years (~104 weeks) - need enough for lag_52
    "M": 730,     # 2 years (~24 months)
    "Y": 1825,    # 5 years
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train all models at all granularities")
    parser.add_argument(
        "--granularities", "-g",
        nargs="+",
        default=GRANULARITIES,
        choices=GRANULARITIES,
        help="Granularities to train (default: all)"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to train (default: all)"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run training in parallel"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="auto",
        choices=["real", "synthetic", "auto"],
        help="Data source: real (UK NESO data), synthetic, or auto (try real first)"
    )
    return parser.parse_args()


def train_model(model: str, granularity: str, days: int, seed: int, source: str = "auto") -> dict:
    """Train a single model at a single granularity."""
    script = MODELS[model]
    script_path = Path(__file__).parent / script

    # Get Python interpreter from the same environment
    python = sys.executable

    cmd = [
        python,
        str(script_path),
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
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per model
        )
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

    # Build list of training tasks
    tasks = []
    for granularity in args.granularities:
        days = DAYS_PER_GRANULARITY[granularity]
        for model in args.models:
            tasks.append((model, granularity, days, args.seed, args.source))

    results = []

    if args.parallel:
        # Run in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(train_model, *task): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                results.append(result)
                status = "OK" if result["success"] else "FAILED"
                print(f"[{status}] {result['model']} @ {result['granularity']}")
    else:
        # Run sequentially
        for model, granularity, days, seed, source in tasks:
            print(f"\n{'=' * 40}")
            print(f"Training {model} @ {granularity} ({days} days, source={source})")
            print("=" * 40)

            result = train_model(model, granularity, days, seed, source)
            results.append(result)

            if result["success"]:
                # Print last few lines of output (metrics)
                lines = result["output"].strip().split("\n")
                for line in lines[-10:]:
                    print(line)
            else:
                print(f"FAILED: {result['error']}")

    # Summary
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
