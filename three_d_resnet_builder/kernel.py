import tensorflow as tf
from tensorflow import keras
from .layers import Conv3DBlock


class TwoPlusOneD(keras.layers.Layer):
    def __init__(self,  kernel_number, kernel_size, strides, padding, use_bn, kernel_regularizer, **kwargs):
        super(TwoPlusOneD, self).__init__(**kwargs)
        self.two = Conv3DBlock(kernel_number, (1, kernel_size, kernel_size), kernel_regularizer, strides, use_bn,
                               padding)
        self.one = Conv3DBlock(kernel_number, (kernel_size, 1, 1), kernel_regularizer, strides, use_bn, padding)

    def __call__(self, inputs, training=None):
        intermediate_output = self.two(inputs)
        return self.one(intermediate_output)


class PThreeDMinusB(keras.layers.Layer):
    def __init__(self,  kernel_number, kernel_size, strides, padding, use_bn, kernel_regularizer, **kwargs):
        super(PThreeDMinusB, self).__init__(**kwargs)
        self.left = Conv3DBlock(kernel_number, (1, kernel_size, kernel_size), kernel_regularizer, strides, use_bn,
                                padding)
        self.right = Conv3DBlock(kernel_number, (kernel_size, 1, 1), kernel_regularizer, strides, use_bn, padding)

    def __call__(self, inputs, training=None):
        left_output = self.left(inputs)
        right_output = self.right(inputs)
        return tf.math.add(left_output, right_output)


class FAST(keras.layers.Layer):
    def __init__(self,  kernel_number, kernel_size, strides, padding, use_bn, kernel_regularizer, **kwargs):
        super(FAST, self).__init__(**kwargs)
        self.xy = Conv3DBlock(kernel_number, (1, kernel_size, kernel_size), kernel_regularizer, strides, use_bn,
                              padding)
        self.xt = Conv3DBlock(kernel_number, (kernel_size, kernel_size, 1), kernel_regularizer, strides, use_bn,
                              padding)
        self.yt = Conv3DBlock(kernel_number, (kernel_size, 1, kernel_size), kernel_regularizer, strides, use_bn,
                              padding)

    def __call__(self, inputs, training=None):
        intermediate_output = self.xy(inputs)
        intermediate_output = self.xt(intermediate_output)
        return self.yt(intermediate_output)


class SplitFAST(tf.keras.layers.Layer):
    def __init__(self,  kernel_number, kernel_size, strides, padding, use_bn, kernel_regularizer, **kwargs):
        super(SplitFAST, self).__init__(**kwargs)
        self.xy = Conv3DBlock(kernel_number, (1, kernel_size, kernel_size), kernel_regularizer, strides, use_bn,
                              padding)
        self.xt = Conv3DBlock(kernel_number, (kernel_size, kernel_size, 1), kernel_regularizer, strides, use_bn,
                              padding)
        self.yt = Conv3DBlock(kernel_number, (kernel_size, 1, kernel_size), kernel_regularizer, strides, use_bn,
                              padding)

    def __call__(self, inputs, training=None):
        intermediate_output = self.xy(inputs)
        left_output = self.xy(intermediate_output)
        right_output = self.xt(intermediate_output)
        return tf.math.add(left_output, right_output)


class ThreeD(keras.layers.Layer):
    def __init__(self,  kernel_number, kernel_size, strides, padding, use_bn, kernel_regularizer, **kwargs):
        super(ThreeD, self).__init__(**kwargs)
        self.three_d = Conv3DBlock(kernel_number, (kernel_size, kernel_size, kernel_size),
                                   kernel_regularizer, strides, use_bn, padding)

    def __call__(self, inputs, training=None):
        return self.three_d(inputs)


def get_kernel_to_name(kernel_type):
    if kernel_type == '3D':
        return ThreeD
    elif kernel_type == '(2+1)D':
        return TwoPlusOneD
    elif kernel_type == 'P3D-B':
        return PThreeDMinusB
    elif kernel_type == 'FAST':
        return FAST
    elif kernel_type == 'split-FAST':
        return SplitFAST
