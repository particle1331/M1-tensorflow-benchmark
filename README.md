# M1-tensorflow-benchmark


I was initially testing if TensorFlow was installed correctly on my M1 such that code automatically runs on the GPU outside any context manager. Since we already have runtimes, it would be interesting to compare local results with those of free GPUs in Kaggle. 🚀

<br>

## MLP Benchmark

<br>

<p align="center">
    <img src=plots/mlp.png width="80%" style="display: block; 
           margin-left: auto;
           margin-right: auto;">
</p>

<br>

Things are looking pretty bad for the M1. Not to mention temps were worrying (~97 °C 🌡). This laptop is fanless after all. 

<br>

## VGG Benchmark

<br>

<p align="center">
    <img src=plots/vgg.png width="80%" style="display: block; 
           margin-left: auto;
           margin-right: auto;">
</p>

<br>

## Running the benchmarks on new compute environments
Running the benchmarks is easy. After setting up your environment run:

```
python run.py <compute> <benchmark>
# ex: python run.py "Kaggle (P100)" mlp
```

Current available benchmark names:
* `mlp`
* `vgg`

This saves a CSV file in `results/<benchmark>/<compute>.csv` containing the results
of the benchmark and automatically updates the plot `plots/<benchmark>.png`. Note that the plot includes all existing results with the pattern `results/<benchmark>/*.csv`.

<br>

## New benchmarks and results

New benchmarks for different architectures and dataset or tasks can be easily created by extending the `Benchmark` abstract class in `benchmarks.py`. See existing implementations in that script for the MLP and VGG architectures.

Results for existing benchmarks can be updated by running benchmarks on different environments, e.g. new RTX cards, M1 Max and M1 Pro.

```
python run.py <new_compute> mlp
```

<br>

## Devices used
- Kaggle P100 and T4 GPU kernels
- Macbook Air 2020 M1 (macOS Monterey v12.1)
