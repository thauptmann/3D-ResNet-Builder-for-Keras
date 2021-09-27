import tensorflow as tf
from tensorflow import keras
from kernel import ThreeD


class ResidualBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, regularizer=None, squeeze_and_excitation=False, strides=1,
                 kernel_type=None, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.squeeze_and_excitation = squeeze_and_excitation
        self.resnet_block = keras.Sequential(
            [
                kernel_type(kernel_number, kernel_size, strides, padding='same', use_bn=True,
                            kernel_regularizer=regularizer),
                kernel_type(kernel_number, kernel_size, strides, padding='same', use_bn=True,
                            kernel_regularizer=regularizer, use_activation=False),
            ]
        )
        self.relu = keras.layers.ReLU()
        if self.squeeze_and_excitation:
            self.se_path = SqueezeAndExcitationPath(kernel_number)

    def __call__(self, inputs, training=None):
        intermediate_output = self.resnet_block(inputs, training=training)
        if self.squeeze_and_excitation:
            weights = self.se_path(intermediate_output, training=training)
            intermediate_output = intermediate_output * weights

        output_sum = tf.add(intermediate_output, inputs)
        output = self.relu(output_sum)
        return output


class ResidualConvBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, regularizer=None, strides=1, squeeze_and_excitation=False,
                 kernel_type=None, **kwargs):
        super(ResidualConvBlock, self).__init__(**kwargs)
        self.squeeze_and_excitation = squeeze_and_excitation
        self.resnet_conv_block = keras.Sequential(
            [
                kernel_type(kernel_number, kernel_size, strides=strides, padding='same', use_bn=True,
                            kernel_regularizer=regularizer),
                kernel_type(kernel_number, kernel_size, strides, padding='same', use_bn=True,
                            kernel_regularizer=regularizer, use_activation=False),
            ]
        )
        self.relu = keras.layers.ReLU()
        self.shortcut_conv = keras.Sequential(
            [
                kernel_type(kernel_number, 1,  kernel_regularizer=regularizer, padding='same',
                            strides=strides, use_bn=True),
                keras.layers.BatchNormalization()
            ]
        )
        if squeeze_and_excitation:
            self.se_path = SqueezeAndExcitationPath(kernel_number)

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_conv_block(inputs, training=training)
        shortcut = self.shortcut_conv(inputs, training=training)

        if self.squeeze_and_excitation:
            weights = self.se_path(intermediate_output, training=training)
            intermediate_output = intermediate_output * weights

        output_sum = tf.add(intermediate_output, shortcut)
        output = self.relu(output_sum)
        return output


class ResidualBottleneckBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, regularizer=None, squeeze_and_excitation=False, strides=1,
                 kernel_type=None, **kwargs):
        super(ResidualBottleneckBlock, self).__init__(**kwargs)
        self.squeeze_and_excitation = squeeze_and_excitation
        self.resnet_bottleneck_block = keras.Sequential(
            [
                ThreeD(kernel_number, 1, strides, 'same', use_bn=True, kernel_regularizer=regularizer),
                kernel_type(kernel_number, kernel_size, kernel_regularizer=regularizer, use_bn=True, strides=strides,
                            padding='same'),
                ThreeD(kernel_number * 4, 1, strides, 'same', use_bn=True, kernel_regularizer=regularizer,
                       use_activation=False),
            ]
        )
        self.relu = keras.layers.ReLU()
        if self.squeeze_and_excitation:
            self.se_path = SqueezeAndExcitationPath(kernel_number * 4)

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_bottleneck_block(inputs, training=training)

        if self.squeeze_and_excitation:
            weights = self.se_path(intermediate_output, training=training)
            intermediate_output = intermediate_output * weights

        output_sum = tf.add(intermediate_output, inputs)
        output = self.relu(output_sum)
        return output


class ResidualConvBottleneckBlock(keras.layers.Layer):
    def __init__(self, kernel_number, kernel_size, regularizer=None, squeeze_and_excitation=False, strides=1,
                 kernel_type=None, **kwargs):
        super(ResidualConvBottleneckBlock, self).__init__(**kwargs)
        self.squeeze_and_excitation = squeeze_and_excitation
        self.resnet_conv_bottleneck_block = keras.Sequential(
            [
                ThreeD(kernel_number, 1, strides, 'same', use_bn=True, kernel_regularizer=regularizer),
                kernel_type(kernel_number, kernel_size, regularizer, padding='same', use_bn=True, strides=strides),
                ThreeD(kernel_number * 4, 1, strides, 'same', use_bn=True, kernel_regularizer=regularizer,
                       use_activation=False),
            ]
        )
        self.relu = keras.layers.ReLU()
        self.shortcut_conv = keras.Sequential([
            kernel_type(kernel_number * 4, 1, strides=strides, kernel_regularizer=regularizer),
            keras.layers.BatchNormalization()
        ]
        )
        if self.squeeze_and_excitation:
            self.se_path = SqueezeAndExcitationPath(kernel_number * 4)

    def call(self, inputs, training=None):
        intermediate_output = self.resnet_conv_bottleneck_block(inputs, training=training)
        shortcut = self.shortcut_conv(inputs, training=training)

        if self.squeeze_and_excitation:
            weights = self.se_path(intermediate_output, training=training)
            intermediate_output = intermediate_output * weights

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

    def __call__(self, inputs, training=None):
        weights = self.se_path(inputs, training=training)
        reshaped_weights = tf.reshape(weights, (-1, 1, 1, 1, self.channel))
        return reshaped_weights


class Conv3DBlock(keras.layers.Layer):
    """
    Convenient layer to create Conv3D -> Batch Normalization -> ReLU blocks faster
    """
    def __init__(self, kernel_number, kernel_size, regularizer=None, strides=1, use_bn=False, padding='same',
                 use_activation=True,  **kwargs):
        super(Conv3DBlock, self).__init__(**kwargs)
        self.custom_conv_3d = keras.Sequential()
        self.custom_conv_3d.add(keras.layers.Conv3D(kernel_number, kernel_size, strides, padding=padding,
                                                    use_bias=not use_bn, kernel_regularizer=regularizer))
        if use_bn:
            self.custom_conv_3d.add(keras.layers.BatchNormalization())
        if use_activation:
            self.custom_conv_3d.add(keras.layers.ReLU())

    def __call__(self, inputs, training=None):
        return self.custom_conv_3d(inputs, training=training)
