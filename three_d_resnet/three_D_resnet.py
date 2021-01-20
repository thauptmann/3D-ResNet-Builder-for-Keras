from tensorflow import keras
from layers import ResidualBlock, ResidualBottleneckBlock, ResidualConvBlock, ResidualConvBottleneckBlock


class ThreeDConvolutionResNet(keras.Model):
    def __init__(self, input_shape, output_shape, kernel_numbers, repetitions, output_activation, regularizer=None,
                 mean=0, std=1, use_bottleneck=False, kernel_size=3, squeeze_and_excitation=False):
        super(ThreeDConvolutionResNet, self).__init__()
        if use_bottleneck:
            residual_conv_block = ResidualConvBottleneckBlock
            residual_block = ResidualConvBlock
        else:
            residual_conv_block = ResidualConvBlock
            residual_block = ResidualBlock

        resnet_head = keras.Sequential([
            keras.layers.Input(input_shape),
            keras.layers.Conv3D(64, 7, 2, use_bias=False, kernel_regularizer=regularizer),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool3D(3, 2)])

        resnet_body = keras.Sequential()
        strides = 1
        for kernel_number, repetition in zip(repetitions, kernel_numbers):
            for i in range(repetition):
                if i == 0:
                    resnet_body.add(residual_conv_block(kernel_number, kernel_size, regularizer,
                                                        squeeze_and_excitation, strides=strides))
                else:
                    resnet_body.add(residual_block(kernel_number, kernel_size, regularizer,
                                                   squeeze_and_excitation, strides=1))
                strides = 2

        # fix resnet tail
        resnet_tail = keras.Sequential(
            [
                keras.layers.GlobalAvgPool3D(),
                keras.layers.Flatten(),
                keras.layers.Dense(output_shape, activation=output_activation),
            ]
        )
        self.resnet = keras.Sequential([resnet_head, resnet_body, resnet_tail])

    def call(self, inputs, training=None, **kwargs):
        return self.resnet(inputs)


# variable 2 layer blocks resnet-18
#        ResidualBlock(64, 3)
#        ResidualBlock(64, 3)

#        ResidualConvBlock(128, 3, 2)
#        ResidualBlock(128, 3)

#        ResidualConvBlock(256, 3, 2)
#        ResidualBlock(256, 3)

#        ResidualConvBlock(512, 3, 2)
#        ResidualBlock(512, 3)

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
