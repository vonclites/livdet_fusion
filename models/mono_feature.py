import tensorflow as tf
from slim.deployment import model_deploy

from iris import models
from iris import networks


def _build(data_provider, params):
    config = params['deploy_config']
    network_params = params['network_params']
    network = networks.catalogue[network_params['network']](network_params)

    with tf.device(config.optimizer_device()):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate']
        )
    deployed_model = model_deploy.deploy(
        config=config,
        model_fn=_clone_fn,
        optimizer=optimizer,
        kwargs={
            'data_provider': data_provider,
            'network': network
        }
    )
    with tf.device(config.variables_device()):
        results = models.assemble_all_output(deployed_model.clones)
    return deployed_model, results


# TODO: Decouple loss from network building.  Stop using model_deploy
def _clone_fn(data_provider,
              network):
    x, y = data_provider.split_batch
    x = list(x.values())[0]

    unscaled_logits, _ = network.build(x, data_provider.is_training)

    mask = tf.not_equal(y['label'], -1)
    with tf.device('/cpu:0'):
        # Needed to mask potentially padded elements of batch
        labels = tf.boolean_mask(y['label'], mask)
        unscaled_logits = tf.boolean_mask(unscaled_logits, mask)

    predictions = tf.argmax(unscaled_logits, 1, name='predictions')
    tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                           logits=unscaled_logits)
    return {
        'predictions': predictions,
        'x': x,
        'labels': labels
    }


def build(data_provider, params):
    with tf.name_scope('model'):
        return _build(data_provider, params)
