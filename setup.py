import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='three_d_resnet_builder',
    version='0.1',
    author='Tony Hauptmann',
    description='A package for generating 3d-resnet models.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/thauptmann/3D-ResNet-for-Keras',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['tensorflow'],
)
