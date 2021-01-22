# 3D-ResNet-for-Keras
A module for creating 3D ResNets based on [1]. It contains convenient functions to build the popular ResNet architectures:
ResNet-18, -34, -52, -102 and -152. It is also possible to create a customised network architecture.

## Installation
Install e.g. with:
~~~shell
$ python -m pip install git+https://github.com/thauptmann/3D-ResNet-for-Keras.git
~~~ 

and import the module with:

~~~python
import three_d_resnet_builder
~~~

## Usage
The convenient functions (*build_three_d_resnet_&ast;*) just need an input shape, an output shape and an activation function to create a network. Additional customisable are the usage of regularization and squeeze-and-excitation layers. By changing *squeeze_and_excitation* to *True* the network will be build with squeeze-and-excitation layers [2]. In the best case, this improves the capabilities of the network with a very small calculation overhead and without increasing the depth of the network.

~~~python
three_d_resnet_builder.build_three_d_resnet_18(input_shape, output_shape, activation_function, regularizer,
                                               squeeze_and_excitation)
~~~


The general function (*build_three_d_resnet*) allows it to change the architecture of the network:
~~~python
three_d_resnet_builder.build_three_d_resnet(input_shape, output_shape, repetitions, output_activation, regularizer, squeeze_and_excitation,
                     use_bottleneck, kernel_size)
~~~


## Demo
For testing purpose I chose the Ucf101 dataset [3]. It contains around 13.000 videos with 101 labels of different actions.

**But caution** this will download approximately 6.5 GiB and will run for a long time, even with a GPU. Without
it, it  will probably crash. (Unfortunately I have not found a smaller and easier example dataset. I am open for advices.)

For decoding the images *ffmpeg* needs to be installed:
~~~shell
$ apt install ffmpeg
~~~

To install the requirements and run the demo, run the following commands at the root directory:
~~~shell
$ python -m pip install -r requirements.txt
$ python demo.py
~~~


---

[1] *K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.*

[2] *J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.*

[3] *K. Soomro, A. Roshan Zamir and M. Shah, "UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild", CRCV-TR-12-01, November, 2012.*
