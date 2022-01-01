# M1-tensorflow-benchmark


TensorFlow (v2.7.0) benchmark results on an M1 Macbook Air 2020 laptop (macOS Monterey v12.1). 

I was initially testing if TensorFlow was installed correctly such that code outside any context automatically runs on the GPU. Since I had GPU timings, it would be interesting to compare runtimes with free GPU services, so I also included Kaggle and Colab in the tests.

<br>

## MLP Benchmark

<br>

<p align="center">
    <img src=plots/mlp_results.png width="80%" style="display: block; 
           margin-left: auto;
           margin-right: auto;">
</p>

<br>

The M1 looks comparable to a K80 which is nice if you always get locked out of Colab (like I do). But temps were worrying (~95 °C) 
this laptop is fanless after all. 🥲 Kaggle's P100 is 4x faster which is expected as the P100 provides 1.6x more GFLOPs and stacks 3x the memory bandwidth of the K80.
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

The name in step 1 can be any short string. This will also be its label in the resulting plot. Make sure to wrap around `""` names if there are spaces. After step 2, a line graph of your results should be added to the above plot in `plots/<BENCHMARK_NAME>_results.png` 🎉 For an example of this process, see `M1.sh`.

<br>

## Contribute
Please contribute by adding more tests with different architectures and dataset, or by running the benchmarks on different environments, e.g. GTX or RTX cards, M1 Max and M1 Pro are very much welcome. 

You can add a new benchmark by extending the `Benchmark` abstract class in `benchmarks.py` you just have to implement a `get_model` method and a `test_indices` attribute. The `get_model` method should return a network for each index in `test_indices`. 

<br>

## Devices used
- Kaggle's P100
- Google Colab's Tesla K80
- Macbook Air 2020 M1 GPU (macOS Monterey v12.1)
- Macbook Air 2020 M1 CPU (macOS Monterey v12.1)