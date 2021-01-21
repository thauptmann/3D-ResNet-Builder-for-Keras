import three_d_resnet_builder
import tensorflow_datasets as tfds
import tensorflow as tf


def train_resnet():
    seed_value = 5
    tf.random.set_seed(seed_value)
    config = tfds.download.DownloadConfig(verify_ssl=False)
    (train_dataset, test_dataset), ds_info = tfds.load("ucf101", split=['train', 'test'], with_info=True,
                                                       shuffle_files=True,
                                                       download_and_prepare_kwargs={"download_config": config},
                                                       batch_size=10)
    input_shape = ds_info.features['video'].shape
    output_shape = ds_info.features['label'].num_classes
    train_dataset = train_dataset.map(lambda sample: normalize_img( sample),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(lambda sample: normalize_img(sample),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    resnet_18 = three_d_resnet_builder.build_three_d_resnet_18(input_shape, output_shape, 'softmax')
    resnet_18.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()],
    )

    resnet_18.fit(train_dataset, epochs=6, validation_data=test_dataset)


def normalize_img(sample):
    """Normalizes images: `uint8` -> `float32`."""
    video = sample['video']
    video = tf.cast(video, tf.float32) / 255.
    return video, sample['label']


if __name__ == '__main__':
    train_resnet()
