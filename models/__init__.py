import collections
import tensorflow as tf
from slim.deployment import model_deploy

import tftools.tools as tft


def configure_deployment(num_gpus):
    return model_deploy.DeploymentConfig(num_clones=num_gpus)


def configure_session():
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=.85)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config)


def assemble_output(clones, key):
    output = tf.concat(
        values=[clone.outputs[key] for clone in clones],
        axis=0
    )
    return output


def assemble_all_output(clones):
    outputs = dict()
    for key in clones[0].outputs.keys():
        outputs[key] = assemble_output(clones, key)
    return outputs


def calculate_accuracies(predictions, labels):
    Accuracies = collections.namedtuple(
        typename='Accuracies',
        field_names=['average',
                     'live',
                     'spoof']
    )
    accuracies, update_op, conf_matrix = tft.per_class_accuracies(
        labels=labels,
        predictions=predictions,
        num_classes=2
    )
    conf_matrix_initializer = tf.variables_initializer(var_list=[conf_matrix])
    accuracies = Accuracies(
        average=tf.reduce_mean(accuracies),
        live=accuracies[0],
        spoof=accuracies[1]
    )
    return accuracies, update_op, conf_matrix_initializer


def warm_start(checkpoint,
               checkpoint_scope,
               model_scope,
               include_patterns,
               exclude_patterns):
    assignment_map = tft.get_warm_start_mapping(
        checkpoint=checkpoint,
        checkpoint_scope=checkpoint_scope,
        model_scope=model_scope,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    tf.train.init_from_checkpoint(
        ckpt_dir_or_file=checkpoint,
        assignment_map=assignment_map
    )
