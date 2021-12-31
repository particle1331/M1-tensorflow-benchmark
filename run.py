import tensorflow as tf
import pandas as pd
import time


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


# MLP benchmark: Total training time for 3 epochs for various depths
def run(id: str) -> dict:
    network_depth = [5, 10, 15, 20]
    results = { depth: {} for depth in network_depth }
    for depth in network_depth:
        model = get_model(depth)
        start_time = time.time()
        model.fit(X_train_scaled, y_train_encoded, epochs=3)
        results[depth][id] = time.time() - start_time

    pd.DataFrame(results).to_csv(f'results/results_{id}.csv', index=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("YOUR_ENV_NAME")
    args = parser.parse_args()

    # Code for particular tests in M1; test on specific devices
    # You can also add tests like this for your special use case.
    if args.mode == 'M1-cpu':
        with tf.device('/CPU:0'):
            run(args.YOUR_ENV_NAME)

    elif args.mode == 'M1-gpu':
        with tf.device('/GPU:0'):
            run(args.YOUR_ENV_NAME)
    
    elif args.mode == 'M1-default':
        with tf.device('/GPU:0'):
            run(args.YOUR_ENV_NAME)

    else:
        # Run benchmark test
        run(args.YOUR_ENV_NAME)