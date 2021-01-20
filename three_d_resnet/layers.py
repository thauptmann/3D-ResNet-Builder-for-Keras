from tensorflow import keras
import tensorflow as tf


class CustomConv3D(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, regularizer, strides=1, use_bn=False, padding='valid',  **kwargs):
        super(CustomConv3D, self).__init__(**kwargs)
        self.custom_conv_3d = keras.Sequential()
        self.custom_conv_3d.add(keras.layers.Conv3D(kernel_number, kernel_size, strides, padding=padding,
                                                    use_bias=not use_bn, kernel_regularizer=regularizer))
        if use_bn:
            self.custom_conv_3d.add(keras.layers.BatchNormalization())
        self.custom_conv_3d.add(keras.layers.ReLU())

    def call(self, inputs, training=None):
        return self.custom_conv_3d(inputs)


class ResidualBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, regularizer, squeeze_and_excitation=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.squeeze_and_excitation = squeeze_and_excitation
        self.resnet_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, kernel_size, 1, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number, kernel_size, padding='same', use_bias=False,
                                    kernel_regularizer=regularizer),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()
        if squeeze_and_excitation:
            self.se_path = SqueezeAndExcitationPath(kernel_number)

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_block(inputs)

        if self.squeeze_and_excitation:
            weights = self.se_path(inputs)
            intermediate_output = intermediate_output * weights

        output_sum = tf.add(intermediate_output, inputs)
        output = self.relu(output_sum)
        return output


class ResidualConvBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, strides=1, squeeze_and_excitation=False, **kwargs):
        super(ResidualConvBlock, self).__init__(**kwargs)
        self.resnet_conv_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, kernel_size, strides=strides, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number, kernel_size, padding='same', use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2()),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()
        self.shortcut_conv = keras.Sequential(
            [
                keras.layers.Conv3D(kernel_number, 1, strides=strides, kernel_regularizer=keras.regularizers.l2()),
                keras.layers.BatchNormalization()
            ]
        )
        if squeeze_and_excitation:
            self.se_path = SqueezeAndExcitationPath(kernel_number)

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_conv_block(inputs)
        shortcut = self.shortcut_conv(inputs)

        if self.squeeze_and_excitation:
            weights = self.se_path(inputs)
            intermediate_output = intermediate_output * weights

        output_sum = tf.add(intermediate_output, shortcut)
        output = self.relu(output_sum)
        return output


class ResidualBottleneckBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, **kwargs):
        super(ResidualBottleneckBlock, self).__init__(**kwargs)
        self.resnet_bottleneck_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, 1, use_bn=True),
                CustomConv3D(kernel_number, kernel_size, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number * 4, 1, use_bias=False),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        intermediate_output = self.resnet_bottleneck_block(inputs)
        output_sum = tf.add(intermediate_output, inputs)
        output = self.relu(output_sum)
        return output


class ResidualConvBottleneckBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, strides=1, **kwargs):
        super(ResidualConvBottleneckBlock, self).__init__(**kwargs)
        self.resnet_conv_bottleneck_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, 1, strides=strides, use_bn=True),
                CustomConv3D(kernel_number, kernel_size, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number * 4, 1, use_bias=False, kernel_regularizer=keras.regularizers.l2()),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()
        self.shortcut_conv = keras.Sequential([
            keras.layers.Conv3D(kernel_number * 4, 1, strides=strides, kernel_regularizer=keras.regularizers.l2()),
            keras.layers.BatchNormalization()
        ]
        )

    def call(self, inputs, training=None, **kwargs):
        intermediate_output = self.resnet_conv_bottleneck_block(inputs)
        shortcut = self.shortcut_conv(inputs)
        output_sum = tf.add(intermediate_output, shortcut)
        output = self.relu(output_sum)
        return output


class SqueezeAndExcitationPath(keras.layers.Layer):
    def __init__(self, channel, ratio=16, **kwargs):
        super(SqueezeAndExcitationPath, self).__init__(**kwargs)
        self.channel = channel
        self.se_path = keras.Sequential(
            [
                keras.layers.GlobalAvgPool3D(),
                keras.layers.Dense(int(self.channel / ratio), activation='relu',
                                   kernel_regularizer=keras.regularizers.l2()),
                keras.layers.Dense(channel, activation='sigmoid'),
            ]
        )

    def call(self, inputs, training=None, **kwargs):
        weights = self.se_path(inputs)
        reshaped_weights = tf.reshape(weights, (-1, 1, 1, 1, self.channel))
        return reshaped_weights


class SqueezeExcitationResidualConvBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, strides=1, **kwargs):
        super(SqueezeExcitationResidualConvBlock, self).__init__(**kwargs)
        self.resnet_conv_block = keras.Sequential(
            [
                CustomConv3D(kernel_number, kernel_size, strides=strides, padding='same', use_bn=True),
                keras.layers.Conv3D(kernel_number, kernel_size, padding='same', use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2()),
                keras.layers.BatchNormalization()
            ]
        )
        self.relu = keras.layers.ReLU()
        self.shortcut_conv = keras.Sequential(
            [
                keras.layers.Conv3D(kernel_number, 1, strides=strides, kernel_regularizer=keras.regularizers.l2()),
                keras.layers.BatchNormalization()
            ])
        self.se_path = SqueezeAndExcitationPath(kernel_number)

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_conv_block(inputs)
        shortcut = self.shortcut_conv(inputs)
        weights = self.se_path(inputs)
        weighted_output = weights * intermediate_output
        output_sum = tf.add(weighted_output, shortcut)
        output = self.relu(output_sum)
        return output
