import three_d_resnet_builder
import tensorflow_datasets as tfds
import tensorflow as tf


def train_resnet():
    seed_value = 5
    tf.random.set_seed(seed_value)
    train_dataset, test_dataset, info = load_ucf101()
    input_shape = info.features['video'].shape
    output_shape = info.features['label'].num_classes

    resnet_18 = three_d_resnet_builder.build_three_d_resnet_18(input_shape, output_shape, 'softmax')
    resnet_18.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top_1_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
        ],
    )

    resnet_18.fit(train_dataset, epochs=5, validation_data=test_dataset)


def load_ucf101():
    autotune = tf.data.experimental.AUTOTUNE
    config = tfds.download.DownloadConfig(verify_ssl=False)
    (train_dataset, test_dataset), ds_info = tfds.load("ucf101", split=['train', 'test'], with_info=True,
                                                       shuffle_files=True, batch_size=10,
                                                       download_and_prepare_kwargs={"download_config": config})
    train_dataset = train_dataset.map(lambda sample: normalize_img(sample),
                                      num_parallel_calls=autotune)
    train_dataset = train_dataset.prefetch(autotune)
    test_dataset = test_dataset.map(lambda sample: normalize_img(sample),
                                    num_parallel_calls=autotune)
    test_dataset = test_dataset.prefetch(autotune)
    return train_dataset, test_dataset, ds_info


def normalize_img(sample):
    video = sample['video']
    """Normalizes images: `uint8` -> `float32`."""
    video = tf.cast(video, tf.float32) / 255.
    return video, sample['label']


if __name__ == '__main__':
    train_resnet()
