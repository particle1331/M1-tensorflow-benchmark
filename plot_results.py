import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import glob

benchmark_xlabels = {
    "mlp": "Network depth",
    "vgg": ""
}

benchmark_metadata = {
    "mlp": {
        "indices_desc": "Network depth",
        "architecture": """
            Training time of an MLP with 5, 10, 15, 20 hidden layers of size 
            1024, ReLU activation, on 50,000 CIFAR-10 images for 3 epochs.
        """,
    },
    "vgg": {
        "indices_desc": "",
        "architecture": """
            Training time of a VGG11/16/19 convolutional 
            network on 50,000 CIFAR-10 images for 3 epochs.
        """,
    }
}


def plot_all_results(benchmark: str):
    dfs = []
    for filename in glob.glob(f'results/{benchmark}*.csv'):
        dfs.append(pd.read_csv(filename, index_col=0))

    result_all = pd.concat(dfs)
    result_all = result_all.sort_values(result_all.columns.max(), ascending=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=200)
    for key in result_all.index:
        ax.plot(result_all.columns.tolist(), result_all.loc[key].values, label=key)

    plt.legend()
    plt.grid()
    txt = "test"
    plt.xlabel(f'''
        {benchmark_metadata[benchmark]["indices_desc"]}

        {benchmark_metadata[benchmark]["architecture"]}
        ''')
    plt.ylabel("Training time (s)")
    plt.tight_layout()
    plt.title(f"""
        TensorFlow {tf.__version__} Benchmark (Macbook Air M1, 2020). Lower is better.
        """)
    plt.savefig(f"plots/{benchmark}_results.png", bbox_inches='tight')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    args = parser.parse_args()

    plot_all_results(args.benchmark)