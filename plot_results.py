import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import glob

dfs = []
for filename in glob.glob('results*.csv'):
    dfs.append(pd.read_csv(filename, index_col=0))

result_all = pd.concat(dfs)
result_all = result_all.sort_values("20", ascending=True)
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
for key in result_all.index:
    ax.plot([5, 10, 15, 20], result_all.loc[key].values, label=key)

plt.legend()
plt.grid()
plt.xlabel("Network depth")
plt.ylabel("Training time (s)")
plt.title(f"TensorFlow {tf.__version__} Benchmark (Macbook Air 2020 M1). Lower is better.")
plt.savefig("results.png")