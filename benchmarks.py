import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty


# Define abstract class for benchmarks. The abstract class should have
#   (1) A `test_indices` list which is the indices for each test
#   (2) A `get_model` method which returns a Keras model for each test index.
class Benchmark(ABC):
    @abstractproperty
    def test_indices(self):
        return NotImplemented

    @abstractmethod
    def get_model(self):
        pass


class MLPBenchmark:
    test_indices = [5, 10, 15, 20]

    # Define deep MLP model constructors
    def get_model(self, depth: int):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
        for _ in range(depth):
            model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


class VGGBenchmark:
    test_indices = ['VGG11', 'VGG16', 'VGG19']

    # Define block constructor
    @staticmethod
    def vgg_block(num_convs, num_channels):
        blk = tf.keras.models.Sequential()
        for _ in range(num_convs):
            blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                        padding='same',activation='relu'))
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

    def get_model(self, test_index: str):
        model = VGGBenchmark.vgg(self.conv_arch[test_index])
        model.compile(
            optimizer='SGD', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model