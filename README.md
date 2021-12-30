# M1-tensorflow-benchmark


TensorFlow (v2.7.0) benchmark results on an M1 Macbook Air 2020 laptop (macOS Monterey v12.1). 

I was initially testing if TensorFlow was installed correctly so that code outside any context manager (`M1 (Default)` in the plot) automatically runs on the GPU (code inside `with tf.device('/GPU:0')` context-manager, `M1 (GPU)` in the plot). It would be interesting to 
compare this with free GPU services, so I also included Kaggle and Colab in the tests. Also tested M1's CPU (`M1 (CPU)` in the plot). 

<br>

<p align="center">
    <img src=results.png width="80%" style="display: block; 
           margin-left: auto;
           margin-right: auto;">
</p>


<br>

This plot shows training time (y-axis) of an MLP with 5, 10, 15, 20 (x-axis) hidden layers of size 1024, and ReLU activation, trained on 50,000 CIFAR-10 images for 3 epochs.


The M1 looks comparable to a K80 which is nice if you always get locked out of Colab (like I do). But temps were worrying (~65 °C) 
this laptop is fanless after all. 🥲 Kaggle's P100 is 4x faster which is expected as the P100 provides 1.6x more GFLOPs and stacks 3x the memory bandwidth of the K80.
This also confirms that the TF installation works 
and that TF code automatically runs on the GPU!


<br>

## Extending the results 

You can extend the results by 
1. Running the relevant parts of the script `run.py` on your environment, and saving the results as `results_<YOUR_ENV_NAME>.csv` the last line of `run.py`.
2. Download the resulting CSV file and save it alongside the other `results_*.csv` files.
3. Then run `plot_results.py`. This extends the above plot `results_plot.png` with another line graph of your results. 🥳

<br>

## Devices used
- Kaggle's P100
- Google Colab's Tesla K80
- Macbook Air 2020 M1 GPU (macOS Monterey v12.1)
- Macbook Air 2020 M1 CPU (macOS Monterey v12.1)

<br>

## Contribute

Please contribute by adding more tests with different architectures and dataset, or by running the same test on different environments, e.g. RTX cards.
