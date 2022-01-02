import tensorflow as tf
import time
from benchmarks import benchmarks


# Get CIFAR10 data; do basic preprocessing
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train_scaled = X_train / 255.0
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')

# Compute training time for training 3 epochs
def time_model_training(model):
    start_time = time.time()
    model.fit(X_train_scaled, y_train_encoded, epochs=3)
    return time.time() - start_time

# Checking existence of benchmark name passed in command line
def get_benchmark(benchmark_name: str):
    try:
        benchmark = benchmarks[benchmark_name]
        return benchmark
    except KeyError:
        raise ValueError(f"Benchmark not found. Choose from: {list(benchmarks.keys())}")
    