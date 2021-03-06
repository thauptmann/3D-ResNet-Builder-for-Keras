from tensorflow import keras
from .layers import ResidualBlock, ResidualBottleneckBlock, ResidualConvBlock, ResidualConvBottleneckBlock


class ThreeDConvolutionResNet(keras.Model):
    """

    """
    def __init__(self, input_shape, output_shape, output_activation, repetitions, regularizer=None,
                 squeeze_and_excitation=False, use_bottleneck=False, kernel_size=3):
        """Build the desired network.

        :param input_shape:
        :param output_shape:
        :param output_activation:
        :param repetitions:
        :param regularizer:
        :param squeeze_and_excitation:
        :param use_bottleneck:
        :param kernel_size:
        """
        super(ThreeDConvolutionResNet, self).__init__()
        if use_bottleneck:
            residual_conv_block = ResidualConvBottleneckBlock
            residual_block = ResidualBottleneckBlock
        else:
            residual_conv_block = ResidualConvBlock
            residual_block = ResidualBlock

        resnet_head = keras.Sequential([
            keras.layers.InputLayer(input_shape),
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
                                                        squeeze_and_excitation=squeeze_and_excitation, strides=strides))
                else:
                    resnet_body.add(residual_block(kernel_number, kernel_size, regularizer,
                                                   squeeze_and_excitation))
                strides = 2
            kernel_number *= 2

        # fix resnet tail
        resnet_tail = keras.Sequential(
            [
                keras.layers.GlobalAvgPool3D(),
                keras.layers.Flatten(),
                keras.layers.Dense(output_shape, activation=output_activation, kernel_regularizer=regularizer),
            ]
        )
        self.resnet = keras.Sequential([resnet_head, resnet_body, resnet_tail])

    def call(self, inputs, training=False, **kwargs):
        """Called to train the network or predict values.

        :param inputs:
        :param training:
        :param kwargs:
        :return:
        """
        return self.resnet(inputs)
