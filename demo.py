import three_d_resnet_builder
import tensorflow_datasets as tfds
import tensorflow as tf


def train_resnet():
    seed_value = 5
    batch_size = 7
    epochs = 100
    scale = 2
    number_of_frames = 100
    tf.random.set_seed(seed_value)
    train_dataset, validation_dataset, test_dataset, info = load_ucf101(batch_size, number_of_frames)
    input_shape = info.features['video'].shape
    width = int(input_shape[1] / scale)
    height = int(input_shape[2] / scale)
    # we convert the rgb images to gray scale
    channels = 1
    input_shape = (None, height, width, channels)
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

    for sample in train_dataset.take(1):
        print(sample[0])
    resnet_18.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)
    results = resnet_18.evaluate(test_dataset, batch_size=batch_size)
    print(f"Results after {epochs} epochs:")
    print('crossentropy, top-1 accuracy, top-5 accuracy', results)


def load_ucf101(batch_size, number_of_frames):
    autotune = tf.data.experimental.AUTOTUNE
    config = tfds.download.DownloadConfig(verify_ssl=False)
    (train_dataset, validation_dataset, test_dataset), ds_info = tfds.load("ucf101", split=['train[:80%]',
                                                                                            'train[80%:]', 'test'],
                                                                           with_info=True, shuffle_files=True,
                                                                           batch_size=batch_size,
                                                                           download_and_prepare_kwargs={
                                                                               "download_config": config})

    train_dataset = train_dataset.map(lambda sample: preprocess_image(sample, number_of_frames),
                                      num_parallel_calls=autotune)
    train_dataset = train_dataset.prefetch(autotune)

    test_dataset = test_dataset.map(lambda sample: preprocess_image(sample, number_of_frames),
                                    num_parallel_calls=autotune)
    test_dataset = test_dataset.prefetch(autotune)

    validation_dataset = validation_dataset.map(lambda sample: preprocess_image(sample, number_of_frames),
                                                num_parallel_calls=autotune)
    validation_dataset = validation_dataset.prefetch(autotune)
    return train_dataset, validation_dataset, test_dataset, ds_info


def preprocess_image(sample, number_of_frames):
    videos = sample['video']
    videos = tf.map_fn(lambda x: tf.image.resize(x, (128, 128)), videos, fn_output_signature=tf.float32)
    converted_videos = tf.image.rgb_to_grayscale(videos)
    converted_videos = tf.cast(converted_videos, tf.float32) / 255.
    return converted_videos[:, :number_of_frames], sample['label']


if __name__ == '__main__':
    train_resnet()
