# M1-tensorflow-benchmark


I was initially testing if TensorFlow was installed correctly on my M1 such that code outside any context automatically runs on the GPU. Since we already have runtimes, it would be interesting to compare local results with those of free GPU services: Kaggle and Colab. 🚀

<br>

## MLP Benchmark

<br>

<p align="center">
    <img src=plots/mlp_results.png width="80%" style="display: block; 
           margin-left: auto;
           margin-right: auto;">
</p>

<br>

The M1 looks comparable to a K80 which is nice if you always get locked out of Colab (like I do). But temps were worrying (~97 °C 🔥) this laptop is fanless after all 🥲. Kaggle's P100 is 4x faster which is expected as the P100 provides 1.6x more GFLOPs and stacks 3x the memory bandwidth of the K80. 

The graph also confirms that the [TF installation works](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-mac-metal-jul-2021.ipynb) 
and that TF code automatically runs on the GPU! 

<br>

## VGG Benchmark

<br>

<p align="center">
    <img src=plots/vgg_results.png width="80%" style="display: block; 
           margin-left: auto;
           margin-right: auto;">
</p>

<br>

## Running the benchmarks on new environments
Running the benchmarks is easy. After setting up your environment do:

1. Run `python run.py <YOUR_ENV_NAME> <BENCHMARK_NAME>`. 
2. Run `plot_results.py <BENCHMARK_NAME>`. 

Current available benchmark names:
* `mlp`
* `vgg`

The environment name in Step 1 can be any short string. This will also be its label in the resulting plot. After Step 2, a line graph of your results should be added to the above plot in `plots/<BENCHMARK_NAME>_results.png` 🎉. See `M1.sh` for the script used to obtain the above plots.


<br>

## Contribute
Please contribute by adding more tests with different architectures and dataset, or by running the benchmarks on different environments, e.g. results with GTX / RTX cards, M1 Max and M1 Pro are very much welcome. 

You can also contribute by extending the `Benchmark` abstract class in `benchmarks.py` to add a new benchmark as follows: 
1. Define a `test_indices` attribute. This can be a sequence of indices corresponding to increasing network complexity. 
2. Define a `get_model` method that returns a compiled model in Keras for each index in `test_indices`. 
3. Add the new benchmark in the `benchmarks` dictionary at the bottom of the script. 
4. Finally, update the metadata dictionary in `plot_results.py` so that plots are informative.
5. Update "Current available benchmark names:" in `README.md` with your new benchmark name.

See existing implementations `benchmarks.VGGBenchmark` and `benchmarks.MLPBenchmark` for inspiration.

<br>

## Devices used
- Kaggle's P100
- Google Colab's Tesla K80
- Macbook Air 2020 M1 (macOS Monterey v12.1)
