import three_d_resnet
import tensorflow_datasets as tfds
import tensorflow as tf


def train_resnet():
    seed_value = 5
    tf.random.set_seed(seed_value)

    (train_dataset, test_dataset), ds_info = tfds.load("ucf101", split=['train', 'test'], with_info=True,
                                                       shuffle_files=True, as_supervised=True)
    input_shape = ds_info.features['image'].shape
    output_shape = ds_info.features['label'].num_classes

    train_dataset = train_dataset.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.batch(30)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(128)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    resnet_18 = three_d_resnet.build_three_d_resnet_18(input_shape, output_shape, 'softmax')
    resnet_18.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    resnet_18.fit(
        train_dataset,
        epochs=6,
        validation_data=test_dataset,
    )


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    train_resnet()
