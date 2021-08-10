import three_d_resnet_builder
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import argparse
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


def train_resnet(use_squeeze_and_excitation, depth, kernel_name, batch_size, number_of_frames, scale):
    seed_value = 5
    epochs = 200
    tf.random.set_seed(seed_value)
    train_dataset, validation_dataset, test_dataset, info = load_ucf101(batch_size, number_of_frames)
    input_shape = info.features['video'].shape
    width = int(input_shape[1] / scale)
    height = int(input_shape[2] / scale)
    # we convert the rgb images to gray scale
    channels = 1
    input_shape = (None, height, width, channels)
    output_shape = info.features['label'].num_classes

    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    resnet = generate_network_architecture(depth, input_shape, output_shape, use_squeeze_and_excitation, kernel_name)

    resnet.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top_1_accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
        ],
    )

    resnet.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=[early_stopping])
    results = resnet.evaluate(test_dataset, batch_size=batch_size)
    print(f"Results after {epochs} epochs:")
    print('Cross entropy, top-1 accuracy, top-5 accuracy', results)


def generate_network_architecture(depth, input_shape, output_shape, use_squeeze_and_excitation, kernel_name):
    if depth == 34:
        resnet = three_d_resnet_builder.build_three_d_resnet_34(input_shape, output_shape, 'softmax', 'l2',
                                                                squeeze_and_excitation=use_squeeze_and_excitation,
                                                                kernel_name=kernel_name)
    elif depth == 50:
        resnet = three_d_resnet_builder.build_three_d_resnet_50(input_shape, output_shape, 'softmax', 'l2',
                                                                squeeze_and_excitation=use_squeeze_and_excitation,
                                                                kernel_name=kernel_name)
    elif depth == 102:
        resnet = three_d_resnet_builder.build_three_d_resnet_102(input_shape, output_shape, 'softmax', 'l2',
                                                                 squeeze_and_excitation=use_squeeze_and_excitation,
                                                                 kernel_name=kernel_name)
    elif depth == 152:
        resnet = three_d_resnet_builder.build_three_d_resnet_152(input_shape, output_shape, 'softmax', 'l2',
                                                                 squeeze_and_excitation=use_squeeze_and_excitation,
                                                                 kernel_name=kernel_name)
    else:
        resnet = three_d_resnet_builder.build_three_d_resnet_18(input_shape, output_shape, 'softmax', 'l2',
                                                                squeeze_and_excitation=use_squeeze_and_excitation,
                                                                kernel_name=kernel_name)
    return resnet


def load_ucf101(batch_size, number_of_frames):
    auto_tune = tf.data.experimental.AUTOTUNE
    config = tfds.download.DownloadConfig(verify_ssl=False)
    (train_dataset, validation_dataset, test_dataset), ds_info = tfds.load("ucf101", split=['train[:80%]',
                                                                                            'train[80%:]', 'test'],
                                                                           with_info=True, shuffle_files=True,
                                                                           batch_size=batch_size,
                                                                           download_and_prepare_kwargs={
                                                                               "download_config": config})

    train_dataset = train_dataset.map(lambda sample: preprocess_image(sample, number_of_frames),
                                      num_parallel_calls=auto_tune)
    train_dataset = train_dataset.prefetch(auto_tune)

    validation_dataset = validation_dataset.map(lambda sample: preprocess_image(sample, number_of_frames),
                                                num_parallel_calls=auto_tune)
    validation_dataset = validation_dataset.prefetch(auto_tune)

    test_dataset = test_dataset.map(lambda sample: preprocess_image(sample, number_of_frames),
                                    num_parallel_calls=auto_tune)
    test_dataset = test_dataset.prefetch(auto_tune)

    return train_dataset, validation_dataset, test_dataset, ds_info


@tf.function
def preprocess_image(sample, number_of_frames):
    videos = sample['video']
    videos = videos[:, :number_of_frames]
    videos = tf.map_fn(lambda x: tf.image.resize(x, (128, 128)), videos, fn_output_signature=tf.float32)
    converted_videos = tf.image.rgb_to_grayscale(videos)
    converted_videos = tf.cast(converted_videos, tf.float32) / 255.
    return converted_videos, sample['label']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_squeeze_and_excitation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--depth', default=18, type=int, choices=[18, 34, 50, 102, 152])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--kernel', default='3D', choices=['3D', '(2+1)D', 'P3D-B', 'FAST', 'split-FAST'])
    args = parser.parse_args()
    train_resnet(args.use_squeeze_and_excitation, args.depth, args.kernel, args.batch_size, args.frames, args.scale)
