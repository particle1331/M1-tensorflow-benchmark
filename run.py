import tensorflow as tf
import pandas as pd
import time
from benchmarks import benchmarks
from utils import get_benchmark, time_model


def run(env_name: str, benchmark_name: str) -> dict:
    benchmark = get_benchmark(benchmark_name)
    results = {depth: {} for depth in benchmark.test_indices}
    for test_index in results.keys():
        model = benchmark.get_model(test_index)
        results[test_index][env_name] = time_model(model)

    pd.DataFrame(results).to_csv(f'results/{benchmark_name}_{env_name}.csv', index=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("benchmark")
    args = parser.parse_args()

    run(args.env_name, args.benchmark)