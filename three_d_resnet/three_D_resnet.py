from tensorflow import keras
from layers import ResidualBlock, ResidualBottleneckBlock, ResidualConvBlock, ResidualConvBottleneckBlock


class ThreeDConvolutionResNet(keras.Model):
    def __init__(self, input_shape, output_shape, repetitions, output_activation, regularizer=None,
                 squeeze_and_excitation=False, use_bottleneck=False, kernel_size=3):
        super(ThreeDConvolutionResNet, self).__init__()
        if use_bottleneck:
            residual_conv_block = ResidualConvBottleneckBlock
            residual_block = ResidualBottleneckBlock
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
        kernel_number = 64
        for repetition in repetitions:
            for i in range(repetition):
                if i == 0:
                    resnet_body.add(residual_conv_block(kernel_number, kernel_size, regularizer,
                                                        squeeze_and_excitation, strides=strides))
                else:
                    resnet_body.add(residual_block(kernel_number, kernel_size, regularizer,
                                                   squeeze_and_excitation, strides=1))
                strides = 2
                kernel_number *= 2

        # fix resnet tail
        resnet_tail = keras.Sequential(
            [
                keras.layers.GlobalAvgPool3D(),
                keras.layers.Flatten(),
                keras.layers.Dense(output_shape, activation=output_activation),
            ]
        )
        self.resnet = keras.Sequential([resnet_head, resnet_body, resnet_tail])

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)
