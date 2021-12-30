import tensorflow as tf

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


if __name__ == "__main__":
    import time
    import pandas as pd
    
    network_depth = [5, 10, 15, 20]
    results = { depth: {} for depth in network_depth }
    for depth in network_depth:

        # Test with CPU
        with tf.device('/CPU:0'):
            cpu_start_time = time.time()
            model_cpu = get_model(depth)
            model_cpu.fit(X_train_scaled, y_train_encoded, epochs=3)
            results[depth]['CPU'] = time.time() - cpu_start_time

        # Test with GPU
        with tf.device('/GPU:0'):
            gpu_start_time = time.time()
            model_gpu = get_model(depth)
            model_gpu.fit(X_train_scaled, y_train_encoded, epochs=3)
            results[depth]['M1'] = time.time() - gpu_start_time

        # Test for code outside any context
        default_start_time = time.time()
        model_default = get_model(depth)
        model_default.fit(X_train_scaled, y_train_encoded, epochs=3)
        results[depth]['Default'] = time.time() - default_start_time

    # Save results
    pd.DataFrame(results).to_csv('results_local.csv', index=True)
