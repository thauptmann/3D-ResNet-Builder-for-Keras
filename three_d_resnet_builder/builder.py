from . import three_D_resnet


def build_three_d_resnet(input_shape, output_shape, repetitions, output_activation, regularizer=None,
                         squeeze_and_excitation=False, use_bottleneck=False, kernel_size=3):
    """Return a full customizable resnet.

    :param input_shape: The input shape of the network as (frames, height, width, channel)
    :param output_shape: The output shape. Dependant on the task of the network.
    :param repetitions: Define the repetitions of the Residual Blocks e.g. (2, 2, 2, 2) for ResNet-18
    :param output_activation: Define the used output activation. Also depends on the task of the network.
    :param regularizer: Define the regularizer to use. E.g. "l1" or "l2"
    :param squeeze_and_excitation: Activate or deactivate SE-Paths.
    :param use_bottleneck: Activate bottleneck layers. Recommended for networks with many layers.
    :param kernel_size: Set the kernel size. Don't need to be changes in almost all cases. It's just exist for
    customization purposes.
    :return: Return the built network.
    """
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, repetitions, output_activation,
                                                  regularizer, squeeze_and_excitation, use_bottleneck, kernel_size)


def build_three_d_resnet_18(input_shape, output_shape, output_activation, regularizer=None,
                            squeeze_and_excitation=False):
    """Return a customizable resnet_18.

    :param input_shape: The input shape of the network as (frames, height, width, channel)
    :param output_shape: The output shape. Dependant on the task of the network.
    :param output_activation: Define the used output activation. Also depends on the task of the network.
    :param regularizer: Defines the regularizer to use. E.g. "l1" or "l2"
    :param squeeze_and_excitation:Activate or deactivate SE-Paths.
    :return: The built ResNet-18
    """
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, output_activation, (2, 2, 2, 2),
                                                  regularizer, squeeze_and_excitation)


def build_three_d_resnet_34(input_shape, output_shape, output_activation, regularizer=None,
                            squeeze_and_excitation=False):
    """Return a customizable resnet_34.

    :param input_shape: The input shape of the network as (frames, height, width, channel)
    :param output_shape: The output shape. Dependant on the task of the network.
    :param output_activation: Define the used output activation. Also depends on the task of the network.
    :param regularizer: Defines the regularizer to use. E.g. "l1" or "l2"
    :param squeeze_and_excitation:Activate or deactivate SE-Paths.
    :return: The built ResNet-34
    """
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, output_activation, (3, 4, 6, 3),
                                                  regularizer, squeeze_and_excitation)


def build_three_d_resnet_50(input_shape, output_shape, output_activation, regularizer=None,
                            squeeze_and_excitation=False):
    """Return a customizable resnet_50.

    :param input_shape: The input shape of the network as (frames, height, width, channels)
    :param output_shape: The output shape. Dependant on the task of the network.
    :param output_activation: Define the used output activation. Also depends on the task of the network.
    :param regularizer: Defines the regularizer to use. E.g. "l1" or "l2"
    :param squeeze_and_excitation:Activate or deactivate SE-Paths.
    :return: The built ResNet-50
    """
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, output_activation, (3, 4, 6, 3),
                                                  regularizer, squeeze_and_excitation, use_bottleneck=True)


def build_three_d_resnet_102(input_shape, output_shape, output_activation, regularizer=None,
                             squeeze_and_excitation=False):
    """Return a customizable resnet_102.

    :param input_shape: The input shape of the network as (frames, height, width, channel)
    :param output_shape: The output shape. Dependant on the task of the network.
    :param output_activation: Define the used output activation. Also depends on the task of the network.
    :param regularizer: Defines the regularizer to use. E.g. "l1" or "l2"
    :param squeeze_and_excitation:Activate or deactivate SE-Paths.
    :return: The built ResNet-102
    """
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, output_activation, (3, 4, 23, 3),
                                                  regularizer, squeeze_and_excitation, use_bottleneck=True)


def build_three_d_resnet_152(input_shape, output_shape, output_activation, regularizer=None,
                             squeeze_and_excitation=False):
    """ Return a customizable resnet_152

    :param input_shape: The input shape of the network as (frames, height, width, channel)
    :param output_shape: The output shape. Dependant on the task of the network.
    :param output_activation: Define the used output activation. Also depends on the task of the network.
    :param regularizer: Defines the regularizer to use. E.g. "l1" or "l2"
    :param squeeze_and_excitation:Activate or deactivate SE-Paths.
    :return: The built ResNet-152
    """
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, output_activation, (3, 8, 36, 3),
                                                  regularizer, squeeze_and_excitation, use_bottleneck=True)
