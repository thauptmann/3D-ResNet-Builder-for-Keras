import three_D_resnet


def build_three_d_resnet(input_shape, output_shape, repetitions, output_activation, regularizer=None,
                         squeeze_and_excitation=False, use_bottleneck=False, kernel_size=3):
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, repetitions, output_activation,
                                                  regularizer, squeeze_and_excitation, use_bottleneck, kernel_size)


def build_three_d_resnet_18(input_shape, output_shape, output_activation, regularizer=None,
                            squeeze_and_excitation=False):
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, (2, 2, 2, 2), output_activation,
                                                  regularizer, squeeze_and_excitation)


def build_three_d_resnet_34(input_shape, output_shape, output_activation, regularizer=None,
                            squeeze_and_excitation=False):
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, (3, 4, 6, 3), output_activation,
                                                  regularizer, squeeze_and_excitation)


def build_three_d_resnet_50(input_shape, output_shape, output_activation, regularizer=None,
                            squeeze_and_excitation=False):
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, (3, 4, 6, 3), output_activation,
                                                  regularizer, squeeze_and_excitation, use_bottleneck=True)


def build_three_d_resnet_102(input_shape, output_shape, output_activation, regularizer=None,
                             squeeze_and_excitation=False):
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, (3, 4, 23, 3), output_activation,
                                                  regularizer, squeeze_and_excitation, use_bottleneck=True)


def build_three_d_resnet_152(input_shape, output_shape, output_activation, regularizer=None,
                             squeeze_and_excitation=False):
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, (3, 8, 36, 3), output_activation,
                                                  regularizer, squeeze_and_excitation, use_bottleneck=True)
