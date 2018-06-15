import tensorflow as tf

from iris import models
from iris import networks
import tftools.tools as tft


def _build(data_provider, params):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params['learning_rate']
    )

    '''Feature Networks'''
    with tf.name_scope('features'):
        feature_outputs = _build_feature_networks(
            data_provider=data_provider,
            params=params['feature_params']
        )

    '''Fusion Network'''
    fusion_outputs = _build_fusion_network(
        feature_embeddings=feature_outputs['embeddings'],
        params=params['fusion_params'],
        is_training=data_provider.is_training
    )

    '''Sparsity Groupings'''
    fused_feature_weights = tf.contrib.framework.get_model_variables(
        scope='{}/fc/fc_1/weights'.format(params['fusion_params']['scope']))[0]

    sparsity_groups = dict()
    begin_index = 0
    feature_group_indices = dict()
    for feature, embedding in feature_outputs['embeddings'].items():
        embedding_size = params['feature_params'][feature]['layer_sizes'][-1]
        feature_group_indices[feature] = (begin_index, embedding_size + begin_index)
        weights = fused_feature_weights[begin_index:embedding_size + begin_index]
        begin_index += embedding_size
        sparsity_groups[feature] = weights

    '''Losses'''
    _, y = data_provider.batch
    with tf.device('/cpu:0'):
        # TODO: Incorporate normal l2 decay losses!
        losses = _calculate_losses(
            fusion_logits=fusion_outputs['unscaled_logits'],
            labels=y['label'],
            sparsity_groups=sparsity_groups,
            sparsity_cost=params['sparsity_cost']
        )

    '''Outputs'''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(
        loss=losses['total'],
        global_step=tf.train.get_global_step()
    )
    train_op = tf.group(train_op, *update_ops)
    model = {
        'train_op': train_op,
        'losses': losses,
        'sparsity_groups': sparsity_groups,
        'fused_feature_weights': fused_feature_weights,
        'fused_feature_indices': feature_group_indices
    }
    results = {
        'predictions': {'features': feature_outputs['predictions'],
                        'fusion': fusion_outputs['predictions']},
        'logits': tf.nn.softmax(fusion_outputs['unscaled_logits']),
        'embeddings': feature_outputs['embeddings'],
        'fused_embeddings': fusion_outputs['fused_embeddings']
    }
    return model, results


def _build_feature_networks(data_provider, params):
    x, _ = data_provider.batch
    embeddings = dict()
    predictions = dict()
    for feature, feature_params in params.items():
        with tf.device(feature_params['device']):
            network = networks.catalogue[feature_params['network']](feature_params)
            unscaled_logits, embedding = network.build(
                x=x[feature_params['tfrecord_key']],
                is_training=data_provider.is_training
            )
            embeddings[feature] = embedding
            predictions[feature] = tf.argmax(unscaled_logits, 1, name='predictions')

    return {'embeddings': embeddings, 'predictions': predictions}


def _build_fusion_network(feature_embeddings, params, is_training):
    network = networks.catalogue[params['network']](params)
    with tf.device(params['device']):
        fused_embeddings = tf.concat(
            values=[embedding for embedding in feature_embeddings.values()],
            axis=1
        )
        unscaled_logits, _ = network.build(
            x=fused_embeddings,
            is_training=is_training
        )
        outputs = {
            'predictions': tf.argmax(unscaled_logits, 1, name='predictions'),
            'unscaled_logits': unscaled_logits,
            'fused_embeddings': fused_embeddings
        }
    return outputs


def _calculate_losses(fusion_logits, labels, sparsity_groups, sparsity_cost):
    with tf.name_scope('loss'):
        # Don't forget sparsity loss is in loss collection already though
        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=fusion_logits,
            scope='cross_entropy'
        )

        sparsity_loss, group_losses = tft.group_sparsity(
            groups=sparsity_groups,
            cost=sparsity_cost
        )
        # Don't add l2 loss of the actual fusion layer
        l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[1:] if l2_losses else [0.0]
        )
        total_loss = tf.add_n(
            inputs=[cross_entropy_loss, sparsity_loss, l2_loss],
            name='total'
        )

    losses = {
        'cross_entropy': cross_entropy_loss,
        'sparsity': {'total': sparsity_loss,
                     'groups': group_losses},
        'l2': l2_loss,
        'total': total_loss
    }
    return losses


def assemble_metrics(data_provider, results):
    with tf.variable_scope('metrics'):
        _, y = data_provider.batch
        labels = y['label']
        with tf.variable_scope('fusion'):
            fusion_acc, fusion_update_op, fusion_init_op = models.calculate_accuracies(
                predictions=results['predictions']['fusion'],
                labels=labels
            )
        feature_accuracies = dict()
        feature_update_ops = []
        feature_init_ops = []
        for feature, predictions in results['predictions']['features'].items():
            with tf.variable_scope(feature):
                acc, update_op, init_op = models.calculate_accuracies(
                    predictions=predictions,
                    labels=labels
                )
                feature_accuracies[feature] = acc
                feature_update_ops.append(update_op)
                feature_init_ops.append(init_op)
        accuracies = {'fusion': fusion_acc,
                      **feature_accuracies}
        update_op = tf.group(fusion_update_op, *feature_update_ops)
        init_op = tf.group(fusion_init_op, *feature_init_ops)
    return accuracies, update_op, init_op


def build(data_provider, params):
    with tf.name_scope('model'):
        return _build(data_provider, params)
