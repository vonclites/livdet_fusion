import os
import json
import numpy as np
import tensorflow as tf

import inputs
from iris import models
from iris.models import fusion
from iris.datasets import configs
import checkmate
import tftools.tools as tft


def _run(params):
    deploy_config = models.configure_deployment(num_gpus=1)
    sess = tf.Session(config=models.configure_session())

    with tf.device(deploy_config.variables_device()):
        global_step = tf.train.create_global_step()

    '''Inputs'''
    data_provider = inputs.DataProvider(
        dataset_params=params['dataset_params'],
        batch_size=params['batch_size'],
        clones_per_batch=deploy_config.num_clones,
        features=params['tfrecord_features'],
        labels=['label'],
        sess=sess
    )

    '''Model'''
    model, output = fusion.build(data_provider, params)

    '''Metrics'''
    accuracies, acc_update_op, acc_init_op = fusion.assemble_metrics(
        data_provider=data_provider,
        results=output
    )

    '''Train Op'''
    train_op = tf.group(model['train_op'], acc_update_op)

    '''Summaries'''
    with tf.name_scope('accuracy'):
        accuracy_summary = tf.summary.merge([
            tf.summary.scalar('avg', accuracies['fusion'].average),
            tf.summary.scalar('live', accuracies['fusion'].live),
            tf.summary.scalar('spoof', accuracies['fusion'].spoof)
        ])

    with tf.name_scope('loss'):
        losses = model['losses']
        tf.summary.scalar('cross_entropy', losses['cross_entropy'])
        tf.summary.scalar('sparsity', losses['sparsity']['total'])
        tf.summary.scalar('l2', losses['l2'])
        tf.summary.scalar('total', losses['total'])

    for feature, loss in losses['sparsity']['groups'].items():
        tf.summary.scalar('group_losses/'+feature, loss)

    for feature, weights in model['sparsity_groups'].items():
        with tf.name_scope('norm_sparsity'):
            normalized_weights = tf.norm(weights, ord='euclidean') \
                                 / tf.to_float(tf.shape(weights)[0])
        tf.summary.scalar('sparsity/'+feature, normalized_weights)
        tf.summary.histogram(feature, weights)
    summary_op = tf.summary.merge_all()

    '''Summary Writers'''
    with tf.device(deploy_config.variables_device()):
        train_writer = tf.summary.FileWriter(params['model_dir'], sess.graph)
        eval_dir = os.path.join(params['model_dir'], 'eval')
        eval_writer = tf.summary.FileWriter(eval_dir, sess.graph)

    '''Model Savers'''
    save_path = os.path.join(params['model_dir'], 'model.ckpt')
    model_saver = tf.train.Saver(
        max_to_keep=params['keep_last_n_checkpoints'],
        save_relative_paths=True
    )
    best_checkpoint_dir = os.path.join(params['model_dir'], 'best')
    best_model_saver = (checkmate.BestCheckpointSaver(
        save_dir=best_checkpoint_dir,
        num_to_keep=params['keep_best_n_checkpoints']
    ) if params['keep_best_n_checkpoints'] else None)

    '''Model Initialization'''
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    last_checkpoint = tf.train.latest_checkpoint(params['model_dir'])
    if last_checkpoint:
        model_saver.restore(sess, last_checkpoint)
    elif 'warm_start_params' in params['fusion_params']:
        # TODO: Update to 1.7, see below
        warm_start_params = params['fusion_params']['warm_start_params']
        op, fd = tf.contrib.framework.assign_from_checkpoint(
            model_path=warm_start_params['checkpoint'],
            var_list=tft.get_warm_start_mapping(**warm_start_params)
        )
        sess.run(op, feed_dict=fd)
    else:
        for feature_params in params['feature_params'].values():
            if 'warm_start_params' in feature_params:
                # TODO: Looks like a new commit has fixed need for this
                # Can instead use models.warm_start(**warm_start_params)
                warm_start_params = feature_params['warm_start_params']
                op, fd = tf.contrib.framework.assign_from_checkpoint(
                    model_path=warm_start_params['checkpoint'],
                    var_list=tft.get_warm_start_mapping(**warm_start_params)
                )
                sess.run(op, feed_dict=fd)

    # TODO: Modularize the hooks
    '''Main Loop'''
    starting_step = sess.run(global_step)
    for train_step in range(starting_step, params['max_train_steps'] + 1):
        sess.run(train_op, feed_dict=data_provider.training_data)

        '''Summary Hook'''
        if train_step % params['summary_interval'] == 0:
            fetches = {'accuracy': accuracies['fusion']}
            fetches['summary'] = accuracy_summary if train_step == 0 else summary_op
            results = sess.run(fetches, feed_dict=data_provider.training_data)
            train_writer.add_summary(results['summary'], train_step)
            print('Train Step {}:  {}'.format(train_step, results['accuracy']))
            sess.run(acc_init_op)

        '''Checkpoint Hooks'''
        if train_step % params['checkpoint_interval'] == 0:
            model_saver.save(sess, save_path, global_step)

        '''Eval Hook'''
        if train_step % params['evaluation_interval'] == 0:
            while True:
                try:
                    sess.run(acc_update_op, feed_dict=data_provider.validation_data)
                except tf.errors.OutOfRangeError:
                    break
            fusion_acc = sess.run(accuracies['fusion'])
            print('Evaluation Step {}:  {}'.format(train_step, fusion_acc))
            fusion_summaries = [
                tf.Summary.Value(tag='accuracy/avg', simple_value=fusion_acc.average),
                tf.Summary.Value(tag='accuracy/live', simple_value=fusion_acc.live),
                tf.Summary.Value(tag='accuracy/spoof', simple_value=fusion_acc.spoof)
            ]
            summary = tf.Summary(value=fusion_summaries)
            eval_writer.add_summary(summary, train_step)

            if best_model_saver:
                best_model_saver.handle(fusion_acc.average, sess, global_step)

            '''Reinitialize'''
            sess.run(acc_init_op)
            data_provider.initialize_validation_data()

            '''Feature Analysis'''
            if train_step % params['feature_eval_interval'] == 0:
                print("Doing feature analysis...")

                '''Retrieve Embeddings'''
                fused_embeddings = []
                while True:
                    try:
                        fused_embeddings.append(sess.run(output['fused_embeddings'],
                                                         data_provider.validation_data))
                    except tf.errors.OutOfRangeError:
                        break
                data_provider.initialize_validation_data()

                '''Analyze Feature'''
                fusion_dim = fused_embeddings[0].shape[1]
                feed_dict = data_provider.validation_data
                feature_summaries = []
                for feature in params['feature_params']:
                    begin_idx, end_idx = model['fused_feature_indices'][feature]

                    '''Retrieve Accuracy'''
                    ignored_indices = np.isin(
                        element=np.arange(0, fusion_dim),
                        test_elements=np.arange(begin_idx, end_idx),
                        invert=True
                    )
                    for embedding_batch in fused_embeddings:
                        fused_embeddings_copy = embedding_batch.copy()
                        fused_embeddings_copy[:, ignored_indices] = 0  # Set all other features to 0
                        feed_dict[output['fused_embeddings']] = fused_embeddings_copy
                        sess.run(acc_update_op, feed_dict=feed_dict)
                    results = sess.run(accuracies)
                    feature_summaries.append(
                        tf.Summary.Value(tag='feature_acc/'+feature,
                                         simple_value=results['fusion'].average)
                    )
                    '''Reinitialize'''
                    sess.run(acc_init_op)
                    data_provider.initialize_validation_data()

                    '''Retrieve Contribution'''
                    for embedding_batch in fused_embeddings:
                        fused_embeddings_copy = embedding_batch.copy()
                        fused_embeddings_copy[:, begin_idx:end_idx] = 0  # Set just this feature to 0
                        feed_dict[output['fused_embeddings']] = fused_embeddings_copy
                        sess.run(acc_update_op, feed_dict=feed_dict)
                    results = sess.run(accuracies)
                    feature_summaries.append(
                        tf.Summary.Value(tag='feature_contrib/' + feature,
                                         simple_value=fusion_acc.average - results['fusion'].average)
                    )
                    '''Reinitialize'''
                    sess.run(acc_init_op)
                    data_provider.initialize_validation_data()
                eval_writer.add_summary(tf.Summary(value=feature_summaries), train_step)
                print("Feature analysis complete")


def run(params):
    configs.write_experiment_params(
        params=params,
        fp=os.path.join(params['model_dir'], 'params.json')
    )

    with tf.Graph().as_default():
        _run(params)
