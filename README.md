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


The M1 looks comparable to a K80 which is nice if you always get locked out of Colab (like I do). But temps were worrying (~65 °C) 
this laptop is fanless after all. 🥲 Kaggle's P100 is 4x faster which is expected as the P100 provides 1.6x more GFLOPs and stacks 3x the memory bandwidth of the K80.
The graph also confirms that the [TF installation works](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-mac-metal-jul-2021.ipynb) 
and that TF code automatically runs on the GPU!


<br>

## Extending the results 

The code for running the benchmarks and consolidating the results in a plot is written so that it can easily incorporate results for new tests. 

1. Run the following script in your environment:
    ```python
    import tensorflow as tf
    import time
    import pandas as pd
    print(tf.__version__)
    
    # Get CIFAR10 data; do basic preprocessing
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train_scaled = X_train / 255.0
    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')

    # Define model constructor
    def get_model(depth):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
        for _ in range(depth):
            model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    YOUR_ENV_NAME = # Your environment's name here.
    network_depth = [5, 10, 15, 20]
    results = { depth: {} for depth in network_depth }
    for depth in network_depth:
        default_start_time = time.time()
        model = get_model(depth)
        model.fit(X_train_scaled, y_train_encoded, epochs=3)
        results[depth][YOUR_ENV_NAME] = time.time() - default_start_time

    # Save results
    pd.DataFrame(results).to_csv(f'results_{YOUR_ENV_NAME}.csv', index=True)
    ```
2. Download the resulting CSV file and save it in the root directory alongside the other `results_*.csv` files.
3. Run `plot_results.py`. Open `results.png`. Another line graph of your results should be added to the above plot. 🥳

<br>

## Devices used
- Kaggle's P100
- Google Colab's Tesla K80
- Macbook Air 2020 M1 GPU (macOS Monterey v12.1)
- Macbook Air 2020 M1 CPU (macOS Monterey v12.1)

<br>

## Contribute

Please contribute by adding more tests with different architectures and dataset, or by running the benchmarks on different environments, e.g. GTX or RTX cards, M1 Max and M1 Pro are very much welcome.
