import os
import tensorflow as tf

import inputs
from iris import models
from iris.models import mono_feature
from iris.datasets import configs
from tftools import tools as tft
import checkmate


def _run(params):
    deploy_config = models.configure_deployment(params['num_gpus'])
    sess = tf.Session(config=models.configure_session())

    with tf.device(deploy_config.variables_device()):
        global_step = tf.train.create_global_step()

    '''Inputs'''
    data_provider = inputs.DataProvider(
        dataset_params=params['dataset_params'],
        batch_size=params['batch_size'],
        clones_per_batch=deploy_config.num_clones,
        features=[params['tfrecord_key']],
        labels=['label'],
        sess=sess
    )

    '''Model'''
    params['deploy_config'] = deploy_config
    model, output = mono_feature.build(data_provider, params)

    '''Metrics'''
    with tf.variable_scope('metrics'):
        accuracies, acc_update_op, acc_init_op = models.calculate_accuracies(
            output['predictions'], output['labels']
        )

    '''Train Op'''
    train_op = tf.group(model.train_op, acc_update_op)

    '''Summaries'''
    with tf.name_scope('accuracy'):
        tf.summary.scalar('avg', accuracies.average)
        tf.summary.scalar('live', accuracies.live)
        tf.summary.scalar('spoof', accuracies.spoof)
    with tf.name_scope('loss'):
        l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                                  model.clones[0].scope))
        tf.summary.scalar('l2', l2_loss)
        tf.summary.scalar('cross_entropy', model.total_loss)
    if params['tfrecord_key'] is 'image':
        tf.summary.image('images', tf.transpose(output['x'], [0, 2, 3, 1]))
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
    elif 'warm_start_params' in params['network_params']:
        warm_start_params = params['network_params']['warm_start_params']
        op, fd = tf.contrib.framework.assign_from_checkpoint(
            model_path=warm_start_params['checkpoint'],
            var_list=tft.get_warm_start_mapping(**warm_start_params)
        )
        sess.run(op, feed_dict=fd)

    '''Main Loop'''
    starting_step = sess.run(global_step)
    for train_step in range(starting_step, params['max_train_steps'] + 1):
        sess.run(train_op, feed_dict=data_provider.training_data)

        '''Summary Hook'''
        if train_step % params['summary_interval'] == 0:
            results = sess.run({'accuracy': accuracies,
                                'summary': summary_op},
                               feed_dict=data_provider.training_data)
            train_writer.add_summary(results['summary'], train_step)
            train_writer.flush()
            print('Train Step {}:  {}'.format(train_step, results['accuracy']))

        '''Checkpoint Hooks'''
        if train_step % params['checkpoint_interval'] == 0:
            model_saver.save(sess, save_path, global_step)

        sess.run(acc_init_op)  # Reinitialize metrics

        '''Eval Hook'''
        if train_step % params['evaluation_interval'] == 0:
            while True:
                try:
                    sess.run(acc_update_op, feed_dict=data_provider.validation_data)
                except tf.errors.OutOfRangeError:
                    break
            results = sess.run(accuracies)

            print('Evaluation Step {}:  {}'.format(train_step, results))
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='accuracy/avg', simple_value=results.average),
                tf.Summary.Value(tag='accuracy/live', simple_value=results.live),
                tf.Summary.Value(tag='accuracy/spoof', simple_value=results.spoof),
            ])
            eval_writer.add_summary(summary, train_step)
            eval_writer.flush()
            # Reinitialize dataset and metrics
            sess.run(acc_init_op)
            data_provider.initialize_validation_data()

            if best_model_saver:
                best_model_saver.handle(results.average, sess, global_step)


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
        'x': tf.boolean_mask(x, mask),
        'labels': labels
    }


def run(params):
    configs.write_experiment_params(
        params=params,
        fp=os.path.join(params['model_dir'], 'params.json')
    )
    with tf.Graph().as_default():
        _run(params)


if __name__ == '__main__':
    FEATURE = 'vgg'

    NETWORK = 'vgg_5'
    WEIGHT_DECAY = 0.001
    BATCH_NORM = 0.95
    DROPOUT = 1.0
    LAYER_SIZE = 100
    EXPERIMENT_NAME = 'feature_analysis/' + FEATURE + '/{}_{}_wd_{}_bn_{}'.format(
        NETWORK, LAYER_SIZE, WEIGHT_DECAY, BATCH_NORM
    )

    VGG_WARM_START_CHECKPOINT = '/home/dom/iris/vgg_16.ckpt'
    VGG_PARAMS = {
        'network': 'vgg',
        'scope': NETWORK,
        'weight_decay': WEIGHT_DECAY,
        'bn_decay': BATCH_NORM,
        'dropout_keep_prob': 1,
        'layer_sizes': [LAYER_SIZE],
        'tfrecord_key': 'image',
        'warm_start_params': {
            'checkpoint': VGG_WARM_START_CHECKPOINT,
            'checkpoint_scope': 'vgg_16',
            'model_scope': NETWORK,
            'include_patterns': [NETWORK],
            'exclude_patterns': ['global_step']
        }
    }

    DATASET_CONFIGS = configs.read_all_dataset_configs('/home/dom/iris/tfrecords')

    trial_sets = {
        # 'iiitd_cogent_0': DATASET_CONFIGS['iiitd_cogent']['iiitd_cogent_0'],
        # 'iiitd_vista_0': DATASET_CONFIGS['iiitd_vista']['iiitd_vista_0'],
        'clarkson2013_0': DATASET_CONFIGS['clarkson2013']['clarkson2013_0']
    }

    BASE_DIR = os.path.join('/mnt/data1/iris_experiments', EXPERIMENT_NAME)

    for dataset_params in trial_sets.values():
        MODEL_DIR = os.path.join(BASE_DIR, dataset_params['dataset'])
        # MODEL_DIR = os.path.join(MODEL_DIR, dataset_params['subset'])

        PARAMS = {
            'model_dir': MODEL_DIR,
            'learning_rate': 0.01,
            'num_gpus': 3,
            'batch_size': 60,
            'dataset_params': dataset_params,
            'network_params': VGG_PARAMS,
            'tfrecord_key': VGG_PARAMS['tfrecord_key'],
            'label': 'label',
            'keep_last_n_checkpoints': 5,
            'keep_best_n_checkpoints': 3,
            'max_train_steps': 2000,
            'summary_interval': 50,
            'checkpoint_interval': 50,
            'evaluation_interval': 50
        }

        run(PARAMS)
