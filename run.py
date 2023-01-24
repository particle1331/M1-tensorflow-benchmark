import glob
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import process_time
from benchmarks import benchmarks


def plot_results(task: str):
    plt.figure(figsize=(7, 6), dpi=200)
    for fn in glob.glob(f'results/{task}/*.csv'):
        df = pd.read_csv(fn)
        x = df.columns.tolist()
        y = df.iloc[0].tolist()
        plt.plot(x, y, linewidth=2, label=fn.split('/')[-1].split('.')[0])
    
    plt.xlabel(f"{benchmarks[task].index_name}")
    plt.ylabel("Run time (s)")
    plt.title(f"{benchmarks[task].description}")
    plt.grid(linestyle='dotted', alpha=0.5)
    plt.legend()
    plt.savefig(f'plots/{task}.png')
    plt.tight_layout()


def time_task(benchmark):
    time_start = process_time()
    benchmark.run()
    time_stop = process_time()
    return time_stop - time_start


def run(compute: str, task: str) -> None:
    if task not in benchmarks.keys():
        raise ValueError(
            f"Task not found. Choose from: {list(benchmarks.keys())}.")
    else:
        bench = benchmarks[task]
        results = {}
        for j in tqdm(bench.index):
            bench.setup(j)
            results[j] = [time_task(bench)]

        # Save results to disk
        pd.DataFrame(results).to_csv(f'results/{task}/{compute}.csv', index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("compute")
    parser.add_argument("task")

    args = parser.parse_args()
    run(compute=args.compute, task=args.task)
    plot_results(task=args.task)
