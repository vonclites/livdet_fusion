import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected, dropout
from tensorflow.contrib.layers import stack, l2_regularizer, batch_norm


class MLP(object):
    name = 'mlp'

    def __init__(self, params):
        self.scope = params['scope']
        self._weight_decay = params['weight_decay']
        self._bn_decay = params.get('bn_decay', None)
        self._dropout_keep_prob = params['dropout_keep_prob']
        self.layer_sizes = params['layer_sizes']
        self._is_built = False
        self.input = None

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.scope, values=[x]):
            with arg_scope(self._arg_scope(is_training)):
                if type(x) is list:
                    x = tf.concat(x, axis=1)
                self.input = x
                return self._mlp(x)

    def _mlp(self, x):
        net = stack(x, fully_connected, self.layer_sizes, scope='fc')
        features = tf.nn.dropout(net,
                                 keep_prob=self._dropout_keep_prob)  # For group dropout
        logits = fully_connected(features,
                                 num_outputs=2,
                                 activation_fn=None,
                                 scope='unscaled_logits')
        return logits, features

    def _arg_scope(self, is_training):
        # TODO: Optional BN probably can be done better
        if self._bn_decay is not None:
            with arg_scope([fully_connected],
                           weights_regularizer=l2_regularizer(self._weight_decay),
                           normalizer_fn=batch_norm,
                           normalizer_params={'is_training': is_training,
                                              'fused': True,
                                              'decay': self._bn_decay}):
                with arg_scope([dropout],
                               keep_prob=self._dropout_keep_prob,
                               noise_shape=[1, 1],
                               is_training=is_training) as sc:
                    return sc
        else:
            with arg_scope([fully_connected],
                           weights_regularizer=l2_regularizer(self._weight_decay)):
                with arg_scope([dropout],
                               keep_prob=self._dropout_keep_prob,
                               noise_shape=[1, 1],
                               is_training=is_training) as sc:
                    return sc
