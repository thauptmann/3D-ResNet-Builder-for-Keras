# 3D-ResNet-for-Keras
A module for creating 3D ResNets based on the work of He et al. [1]. It contains convenient functions to build the popular ResNet architectures:
ResNet-18, -34, -52, -102 and -152. It is also possible to create  customised network architectures.

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

The input shape has to be (frames, height, width, channels)

~~~python
three_d_resnet_builder.build_three_d_resnet_18(input_shape, output_shape, activation_function, regularizer,
                                               squeeze_and_excitation, kernel_name)
~~~


The general function (*build_three_d_resnet*) allows it to change the architecture of the network:
~~~python
three_d_resnet_builder.build_three_d_resnet(input_shape, output_shape, repetitions, output_activation, regularizer, squeeze_and_excitation,
                     use_bottleneck, kernel_size, kernel_name)
~~~


## Kernel
The package contains different types of kernel. The type can be choosen with help of the *kernel_name* variable in the build function. Possible kernels are: '3D', '(2+1)D'[4], 'P3D-B'[5], 'FAST'[6], 'split-FAST'[6].

## Demo
For testing purposes I chose the Ucf101 dataset [3] to train on. It contains approximately 13.000 videos with 101 labels of different actions. The purpose of the demo is to show how to use the package, not to get the best results. Therefore important preprocessing steps (e.g. shuffling and data augmentation) are missing.

**Caveat:** This will download approximately 6.5 GiB and will run for several hours on a GPU. Without
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

### Results

| Model | Top-1 Accuracy | Top-5-Accuracy |
|---|---|---|
|ResNet-18|%|%|
|SE-ResNet-18|%|%|


---

[1] *K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.*

[2] *J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.*

[3] *K. Soomro, A. Roshan Zamir and M. Shah, "UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild", CRCV-TR-12-01, November, 2012.*

[4] *D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri. A closer look at spatiotemporal convolutions for action recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 6450–6459. (IEEE), 2018.* 

[5] *Z. Qiu, T. Yao, and T. Mei. Learning spatio-temporal representation with pseudo-3d residual networks. In International Conference on Computer Vision (ICCV), pages 5534–5542. (IEEE), 2017.*

[6] *A. Stergiou and R. Poppe, “Spatio-Temporal FAST 3D Convolutions for Human Action Recognition,” presented at the 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), Dec. 2019. doi: 10.1109/icmla.2019.00036.*
