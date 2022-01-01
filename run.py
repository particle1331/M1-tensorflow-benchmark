import tensorflow as tf
import pandas as pd
import time
import benchmarks


# Get CIFAR10 data; do basic preprocessing
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train_scaled = X_train / 255.0
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')


# Total training time for 3 epochs for increasing complexity
def run(env_name: str, benchmark: str) -> dict:
    
    # Choose from implemented benchmarks
    implemented_benchmarks = {
        'mlp': benchmarks.MLPBenchmark(),
        'vgg': benchmarks.VGGBenchmark()
    }
    try:
        benchmark = implemented_benchmarks[benchmark]
    except KeyError:
        raise ValueError(f"Benchmark not found. Choose from:\n{implemented_benchmarks.keys()}")

    results = {depth: {} for depth in benchmark.test_indices}
    for test_id in results.keys():
        model = benchmark.get_model(test_id)
        start_time = time.time()
        model.fit(X_train_scaled, y_train_encoded, epochs=3)
        results[test_id][env_name] = time.time() - start_time

    pd.DataFrame(results).to_csv(f'results/{benchmark}_{env_name}.csv', index=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("benchmark")
    args = parser.parse_args()

    run(args.env_name, args.benchmark)