from tensorflow import keras
from models.custom_layers import SqueezeAndExcitationResidualBlock, SqueezeExcitationResidualConvBlock


class ThreeDConvolutionSqueezeAndExciationResNet18(keras.Model):
    def __init__(self, width, height, frames, channels, mean, std, output=1):
        super(ThreeDConvolutionSqueezeAndExciationResNet18, self).__init__()
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
            SqueezeAndExcitationResidualBlock(64, 3),
            SqueezeAndExcitationResidualBlock(64, 3),

            SqueezeExcitationResidualConvBlock(128, 3, 2),
            SqueezeAndExcitationResidualBlock(128, 3),

            SqueezeExcitationResidualConvBlock(256, 3, 2),
            SqueezeAndExcitationResidualBlock(256, 3),

            SqueezeExcitationResidualConvBlock(512, 3, 2),
            SqueezeAndExcitationResidualBlock(512, 3),

            # fix resnet tail
            keras.layers.GlobalAvgPool3D(),
            keras.layers.Flatten(),
            keras.layers.Dense(output),
        ]
        )

    def call(self, inputs, training=False, **kwargs):
        return self.resnet(inputs)
