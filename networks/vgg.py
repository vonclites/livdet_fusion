import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, stack, repeat
from tensorflow.contrib.layers import l2_regularizer, batch_norm, dropout, flatten
from tensorflow.contrib.framework import arg_scope


# TODO: Allow for multiple final FC layers
class VGG(object):
    name = 'vgg'
    versions = [
        'vgg_5',
        'vgg_5b',
        'vgg_8',
        'vgg_11'
        ]

    def __init__(self, params):
        self.version = params['scope']
        self._weight_decay = params['weight_decay']
        self._bn_decay = params['bn_decay']
        self._dropout_keep_prob = params['dropout_keep_prob']
        self.layer_sizes = params['layer_sizes']
        self._num_classes = 2
        self._is_built = False
        arch_map = {
            'vgg_5': self._vgg_5,
            'vgg_8': self._vgg_8,
            'vgg_11': self._vgg_11
        }
        self.architecture = arch_map[self.version]

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.version, values=[x]):
            with arg_scope(self._arg_scope(is_training)):
                return self.architecture(x)

    def _vgg_5(self, images):
        net = repeat(images, 2, conv2d, 64, [3, 3], scope='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool1')
        net = repeat(net, 2, conv2d, 128, [3, 3], scope='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool2')
        net = flatten(net, scope='flatten')
        net = stack(net, fully_connected, self.layer_sizes, scope='fc3')
        features = dropout(net, scope='drop3')
        logits = fully_connected(net, self._num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='unscaled_logits')
        return logits, features

    def _vgg_8(self, x):
        net = repeat(x, 2, conv2d, 64, [3, 3], scope='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool1')
        net = repeat(net, 2, conv2d, 128, [3, 3], scope='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool2')
        net = repeat(net, 3, conv2d, 256, [3, 3], scope='conv3')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool3')
        net = tf.layers.flatten(net, name='flatten')
        net = stack(net, fully_connected, self.layer_sizes, scope='fc4')
        features = dropout(net, scope='drop4', noise_shape=[1, 1])
        logits = fully_connected(features, self._num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='unscaled_logits')
        return logits, features

    def _vgg_11(self, x):
        net = repeat(x, 2, conv2d, 64, [3, 3], scope='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool1')
        net = repeat(net, 2, conv2d, 128, [3, 3], scope='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool2')
        net = repeat(net, 3, conv2d, 256, [3, 3], scope='conv3')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool3')
        net = repeat(net, 3, conv2d, 512, [3, 3], scope='conv4')
        net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                      data_format='channels_first', name='pool4')
        net = tf.layers.flatten(net, name='flatten')
        net = stack(net, fully_connected, self.layer_sizes, scope='fc5')
        features = dropout(net, scope='drop5')
        logits = fully_connected(features, self._num_classes,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='unscaled_logits')
        return logits, features

    def _arg_scope(self, is_training):
        with arg_scope([conv2d, fully_connected],
                       weights_regularizer=l2_regularizer(self._weight_decay),
                       normalizer_fn=batch_norm,
                       normalizer_params={'is_training': is_training,
                                          'fused': True,
                                          'decay': self._bn_decay}
                       ):
            with arg_scope([conv2d, batch_norm], data_format='NCHW'):
                with arg_scope([dropout],
                               keep_prob=self._dropout_keep_prob,
                               is_training=is_training):
                    with arg_scope([conv2d],
                                   outputs_collections='feature_maps') \
                            as sc:
                        return sc
