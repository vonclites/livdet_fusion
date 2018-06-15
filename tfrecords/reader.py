import numpy as np
import tensorflow as tf

import inputs


def read_examples(tfrecords, feature, means_file=None):
    sess = tf.Session(config=_configure_session())

    dataset = inputs.ValidationPipeline(
        files=tfrecords,
        batch_size=100,
        scaler_file=means_file
    ).build()
    examples, labels = dataset.make_one_shot_iterator().get_next()
    examples = examples[feature]
    labels = labels['label']

    not_padded = tf.not_equal(labels, -1)
    labels = tf.boolean_mask(labels, not_padded)
    examples = tf.boolean_mask(examples, not_padded)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    feature_sizes = {
        'sid': 600,
        'daisy': 600,
        'lbp': 9,
        'hog': 200,
        'coa_lbp': 3072,
        'bsif': 256
    }

    x = np.empty(shape=[0, feature_sizes[feature]])
    y = np.array([])

    while True:
        try:
            example_batch, label_batch = sess.run([examples, labels])
            x = np.append(x, example_batch, axis=0)
            y = np.append(y, label_batch)
        except tf.errors.OutOfRangeError:
            break
    return x, y


def _configure_session():
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=.9)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config)
