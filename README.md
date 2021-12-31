# M1-tensorflow-benchmark


TensorFlow (v2.7.0) benchmark results on an M1 Macbook Air 2020 laptop (macOS Monterey v12.1). 

I was initially testing if TensorFlow was installed correctly so that code outside any context manager automatically runs on the GPU by using the `with tf.device('/GPU:0')` context manager. It would be interesting to 
compare this with free GPU services, so I also included Kaggle and Colab in the tests. Also tested M1's CPU. 

<br>

<p align="center">
    <img src=results.png width="80%" style="display: block; 
           margin-left: auto;
           margin-right: auto;">
</p>


<br>

This plot shows training time (y-axis) of an MLP with 5, 10, 15, 20 (x-axis) hidden layers of size 1024, and ReLU activation, trained on 50,000 CIFAR-10 images for 3 epochs.


The M1 looks comparable to a K80 which is nice if you always get locked out of Colab (like I do). But temps were worrying (~95 °C) 
this laptop is fanless after all. 🥲 Kaggle's P100 is 4x faster which is expected as the P100 provides 1.6x more GFLOPs and stacks 3x the memory bandwidth of the K80.
The graph also confirms that the [TF installation works](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-mac-metal-jul-2021.ipynb) 
and that TF code automatically runs on the GPU!


<br>

## Extending the results 

Adding new benchmark results to the above plot is easy:

1. Run `python run.py <YOUR_ENV_NAME>`. 
2. Run `plot_results.py`. Open `results.png`. A line graph of your results should be added to the above plot. 🎉

The name in step 1 can be any short string. This will also be its label in the resulting plot. Make sure to wrap around `""` if there are spaces. The resulting CSV file should be alongside the other `results_*.csv` files for step 2 to work.

<br>

## Devices used
- Kaggle's P100
- Google Colab's Tesla K80
- Macbook Air 2020 M1 GPU (macOS Monterey v12.1)
- Macbook Air 2020 M1 CPU (macOS Monterey v12.1)

<br>

## Contribute

Please contribute by adding more tests with different architectures and dataset, or by running the benchmarks on different environments, e.g. GTX or RTX cards, M1 Max and M1 Pro are very much welcome.
