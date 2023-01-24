import tensorflow as tf
from pathlib import Path
from tensorflow.keras.datasets import cifar10
from abc import ABC, abstractmethod, abstractproperty


class Benchmark(ABC):
    @abstractproperty
    def index(self) -> list:
        pass
    
    @abstractproperty
    def description(self) -> str:
        pass

    @abstractproperty
    def index_name(self) -> str:
        pass

    @abstractmethod
    def setup(self, index: int):
        pass

    @abstractmethod
    def run(self):
        pass


# Global variables for benchmarks 
(X, y), _ = cifar10.load_data()
X = X / 255.0


class MLPBenchmark:
    index = [5, 10, 15, 20]
    index_name = "No. of hidden layers"
    description = """
        MLP (1024 hidden layer width + ReLU) training with 
        Keras fit default args for 3 epochs on 50k CIFAR-10 images
    """

    def setup(self, depth: int):
        layers  = [tf.keras.layers.Flatten(input_shape=(32, 32, 3))] 
        layers += [tf.keras.layers.Dense(1024, activation='relu') for _ in range(depth)]
        layers += [tf.keras.layers.Dense(10, activation='sigmoid')]
        
        self.model = tf.keras.Sequential(layers)
        self.model.compile(
            optimizer='SGD', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

    def run(self):
        self.model.fit(X, y, epochs=3)


class VGGBenchmark:
    index = ['VGG11', 'VGG16', 'VGG19']
    index_name = ''
    description = """
        VGG training on 50k CIFAR-10 images 
        with Keras fit default args for 3 epochs 
    """

    @staticmethod
    def vgg_block(num_convs, num_channels):
        blk = tf.keras.models.Sequential()
        for _ in range(num_convs):
            blk.add(tf.keras.layers.Conv2D(
                    num_channels,
                    kernel_size=3,
                    padding='same',
                    activation='relu'
                )
            )
        blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        return blk

    # Define model constructor
    @staticmethod
    def vgg(conv_arch):
        net = tf.keras.models.Sequential()
        
        # The convolutional part
        for (num_convs, num_channels) in conv_arch:
            net.add(VGGBenchmark.vgg_block(num_convs, num_channels))
        
        # The fully-connected part
        net.add(tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)]))
        return net

    # https://www.geeksforgeeks.org/vgg-16-cnn-model/
    conv_arch = {
        'VGG11': ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
        'VGG16': ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
        'VGG19': ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512)),
    }

    def setup(self, index: str):
        self.model = VGGBenchmark.vgg(self.conv_arch[index])
        self.model.compile(
            optimizer='SGD', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

    def run(self):
        self.model.fit(X, y, epochs=3)


# Feel free to extend the collection of benchmarks!
benchmarks = {
    "mlp": MLPBenchmark(),
    "vgg": VGGBenchmark(),
    # ...
}

# Create directories for saving benchmarks
Path("results").mkdir(exist_ok=True)
for k in benchmarks.keys():
    Path(f"results/{k}").mkdir(exist_ok=True)
