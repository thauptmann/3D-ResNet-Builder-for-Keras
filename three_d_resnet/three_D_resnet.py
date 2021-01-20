from tensorflow import keras
from models.custom_layers import ResidualBlock, ResidualBottleneckBlock, ResidualConvBlock, ResidualConvBottleneckBlock


class ThreeDConvolutionResNet18(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output=1):
        super(ThreeDConvolutionResNet18, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=keras.regularizers.l2()),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 2 layer blocks
            ResidualBlock(64, 3),
            ResidualBlock(64, 3),

            ResidualConvBlock(128, 3, 2),
            ResidualBlock(128, 3),

           #  ResidualConvBlock(256, 3, 2),
           #  ResidualBlock(256, 3),

            # ResidualConvBlock(512, 3, 2),
            # ResidualBlock(512, 3),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)


class ThreeDConvolutionResNet34(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output=1):
        super(ThreeDConvolutionResNet34, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=keras.regularizers.l2()),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 2 layer blocks
            ResidualBlock(64, 3),
            ResidualBlock(64, 3),
            ResidualBlock(64, 3),

            ResidualConvBlock(128, 3, 2),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),

            ResidualConvBlock(256, 2),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),

            ResidualConvBlock(512, 3, 2),
            ResidualBlock(512, 3),
            ResidualBlock(512, 3),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)


class ThreeDConvolutionResNet50(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output=1):
        super(ThreeDConvolutionResNet50, self).__init__()
        input_shape = (frames, width, height, channels)
        self.resnet = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(scale=1 / std, offset=-mean / std,
                                                              input_shape=input_shape),
            # fix resnet head
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=keras.regularizers.l2()),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2),

            # variable 3 layer blocks
            ResidualConvBottleneckBlock(64, 3, 2),
            ResidualBottleneckBlock(64, 3),
            ResidualBottleneckBlock(64, 3),

            ResidualConvBottleneckBlock(128, 3, 2),
            ResidualBottleneckBlock(128, 3),
            ResidualBottleneckBlock(128, 3),
            ResidualBottleneckBlock(128, 3),

            ResidualConvBottleneckBlock(256, 3, 2),
            ResidualBottleneckBlock(256, 3),
            ResidualBottleneckBlock(256, 3),
            ResidualBottleneckBlock(256, 3),
            ResidualBottleneckBlock(256, 3),
            ResidualBottleneckBlock(256, 3),

            ResidualConvBottleneckBlock(512, 3, 2),
            ResidualBottleneckBlock(512, 3),
            ResidualBottleneckBlock(512, 3),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)
