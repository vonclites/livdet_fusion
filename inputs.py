import numpy as np

import tensorflow as tf


class DataProvider(object):
    def __init__(self,
                 dataset_params,
                 batch_size,
                 clones_per_batch,
                 features,
                 labels,
                 sess):
        assert batch_size % 2 == 0
        batch_size = batch_size * clones_per_batch

        self._sess = sess
        self._split_index_iter = iter(range(clones_per_batch))

        if 'train_live' in dataset_params and 'train_spoof' in dataset_params:
            training_dataset = TrainingPipeline(
                live_files=dataset_params['train_live'],
                spoof_files=dataset_params['train_spoof'],
                batch_size=batch_size,
                scaler_file=dataset_params.get('scaler_file', None)
            ).build()
        else:
            training_dataset = None

        if 'test' in dataset_params:
            validation_dataset = ValidationPipeline(
                files=dataset_params['test'],
                batch_size=batch_size,
                scaler_file=dataset_params.get('scaler_file', None)
            ).build()
        else:
            validation_dataset = None

        with tf.name_scope('data_provider'), tf.device('/cpu:0'):
            self._handle = tf.placeholder(tf.string, shape=[])
            self._is_training = tf.placeholder(tf.bool, [], 'is_training')

            iterator = tf.data.Iterator.from_string_handle(
                string_handle=self._handle,
                output_types=training_dataset.output_types,
                output_shapes=training_dataset.output_shapes
            )

            if training_dataset:
                training_iterator = training_dataset.make_one_shot_iterator()
                self._training_handle = sess.run(training_iterator.string_handle())

            if validation_dataset:
                self.validation_iterator = validation_dataset.make_initializable_iterator()
                self.initialize_validation_data()
                self._validation_handle = sess.run(self.validation_iterator.string_handle())

            x, y = iterator.get_next()

            # Filter requested tensors
            x = {feature: values
                 for feature, values in x.items()
                 if feature in features}
            y = {label: value
                 for label, value in y.items()
                 if label in labels}

            # Needed to mask potentially padded elements of batch
            mask = tf.not_equal(y['label'], -1)

            self._x = {feature: tf.boolean_mask(values, mask)
                       for feature, values in x.items()}
            self._y = {label: tf.boolean_mask(value, mask)
                       for label, value in y.items()}

            self._x_splits = dict()
            for feature, values in x.items():
                self._x_splits[feature] = tf.split(
                    value=values,
                    num_or_size_splits=clones_per_batch,
                    name='x_splits'
                )

            self._y_splits = dict()
            for label, value in y.items():
                self._y_splits[label] = tf.split(
                    value=value,
                    num_or_size_splits=clones_per_batch,
                    name='y_splits'
                )

    @property
    def split_batch(self):
        split_index = next(self._split_index_iter)
        x = {feature: values[split_index]
             for feature, values in self._x_splits.items()}
        y = {label: value[split_index]
             for label, value in self._y_splits.items()}
        return x, y

    @property
    def batch(self):
        return self._x, self._y

    @property
    def is_training(self):
        return self._is_training

    @property
    def training_data(self):
        return {self._handle: self._training_handle,
                self._is_training: True}

    @property
    def validation_data(self):
        return {self._handle: self._validation_handle,
                self._is_training: False}

    def initialize_validation_data(self):
        self._sess.run(self.validation_iterator.initializer)


class Pipeline(object):
    # Mapping from feature names to data types for TFRecord parser
    _DEFAULT_FEATURE_PARSERS = {
        'image': tf.FixedLenFeature([], tf.string),
        'lbp': tf.FixedLenFeature([9], tf.int64),
        'hog': tf.FixedLenFeature([200], tf.float32),
        'coa_lbp': tf.FixedLenFeature([3072], tf.float32),
        'bsif': tf.FixedLenFeature([256], tf.float32),
        'daisy': tf.FixedLenFeature([600], tf.int64),
        'sid': tf.FixedLenFeature([600], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'dataset': tf.FixedLenFeature([], tf.int64),
    }

    # Mapping from label names to data types for TFRecord parser
    _DEFAULT_LABEL_PARSERS = {
        'label': tf.FixedLenFeature([], tf.int64),
        'id': tf.FixedLenFeature([], tf.int64),
        'contacts': tf.FixedLenFeature([], tf.int64),
        'type': tf.FixedLenFeature([], tf.int64),
        'dataset': tf.FixedLenFeature([], tf.int64),
    }

    # Mapping from label names to preprocessing functions
    _DEFAULT_LABEL_TRANSFORMS = dict()

    @staticmethod
    def _create_preprocess_image_fn(
            distort,
            central_crop=0.6,
            image_height=224,
            image_width=224):
        def preprocess_image(image):
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize_images(image, [image_height, image_width])
            image = tf.image.convert_image_dtype(image, tf.float32)
            if distort:
                image = tf.image.random_flip_left_right(image)
            if central_crop != 1:
                image = tf.image.central_crop(image, central_crop)
                fraction_offset = int(1 / ((1 - central_crop) / 2.0))
                bbox_h_start = int(image_height / fraction_offset)
                bbox_w_start = int(image_width / fraction_offset)
                bbox_h_size = image_height - bbox_h_start * 2
                bbox_w_size = image_width - bbox_w_start * 2
                assert bbox_h_size > 0 and bbox_w_size > 0
                image.set_shape((bbox_h_size, bbox_w_size, 3))
            image = tf.transpose(image, [2, 0, 1])
            return image
        return preprocess_image

    @staticmethod
    def _create_preprocess_fn(
            feature_parsers,
            feature_transforms,
            label_parsers,
            label_transforms):
        for feature in feature_transforms.keys():
            assert feature in feature_parsers
        for label in label_transforms.keys():
            assert label in label_parsers

        def preprocess_fn(serialized_example):
            features = tf.parse_single_example(serialized_example, feature_parsers)
            labels = tf.parse_single_example(serialized_example, label_parsers)

            # Apply the given transforms to each feature and label
            for feature, transform in feature_transforms.items():
                features[feature] = transform(features[feature])
            for label, transform in label_transforms.items():
                labels[label] = transform(labels[label])
            return features, labels

        return preprocess_fn


class TrainingPipeline(Pipeline):
    _DEFAULT_FEATURE_TRANSFORMS = {
            'image': Pipeline._create_preprocess_image_fn(distort=True),
            'lbp': tf.to_float,
            'daisy': tf.to_float,
            'sid': tf.to_float
        }

    def __init__(self,
                 live_files,
                 spoof_files,
                 batch_size,
                 repeat=None,
                 threads=5,
                 shuffle_buffer=1000,
                 feature_parsers=None,
                 feature_transforms=None,
                 label_parsers=None,
                 label_transforms=None,
                 scaler_file=None,
                 seed=None):
        assert batch_size % 2 == 0

        self.live_files = live_files
        self.spoof_files = spoof_files
        self.batch_size = batch_size
        self.repeat = repeat
        self.threads = threads
        self.shuffle_buffer = shuffle_buffer
        self.feature_parsers = feature_parsers
        self.label_parsers = label_parsers
        self.label_transforms = label_transforms
        self.seed = seed

        if feature_transforms:
            self.feature_transforms = feature_transforms
        elif scaler_file:
            scalers = np.load(scaler_file).item()
            self.feature_transforms = {
                'image': Pipeline._create_preprocess_image_fn(distort=True),
                'lbp': lambda x: tf.divide(tf.subtract(tf.to_float(x), scalers['lbp'].mean_),
                                           scalers['lbp'].scale_),
                'daisy': lambda x: tf.divide(tf.subtract(tf.to_float(x), scalers['daisy'].mean_),
                                             scalers['daisy'].scale_),
                'sid': lambda x: tf.divide(tf.subtract(tf.to_float(x), scalers['sid'].mean_),
                                           scalers['sid'].scale_),
                'hog': lambda x: tf.divide(tf.subtract(x, scalers['hog'].mean_),
                                           scalers['hog'].scale_),
                'bsif': lambda x: tf.divide(tf.subtract(x, scalers['bsif'].mean_),
                                            scalers['bsif'].scale_),
                'coa_lbp': lambda x: tf.divide(tf.subtract(x, scalers['coa_lbp'].mean_),
                                               scalers['coa_lbp'].scale_)
            }
        else:
            self.feature_transforms = self._DEFAULT_FEATURE_TRANSFORMS

    def build(self):
        with tf.name_scope('training_data'):
            # TODO: Refactor into parent
            preprocess_fn = self._create_preprocess_fn(
                feature_parsers=(self.feature_parsers or
                                 self._DEFAULT_FEATURE_PARSERS),
                feature_transforms=self.feature_transforms,
                label_parsers=(self.label_parsers or
                               self._DEFAULT_LABEL_PARSERS),
                label_transforms=(self.label_transforms or
                                  self._DEFAULT_LABEL_TRANSFORMS)
            )

            def _interleave_map_fn(files):
                ds = tf.data.TFRecordDataset(files)
                ds = ds.map(
                    map_func=preprocess_fn,
                    num_parallel_calls=self.threads
                )
                ds = ds.repeat(self.repeat)
                ds = ds.shuffle(
                    buffer_size=self.shuffle_buffer,
                    seed=self.seed
                )
                return ds

            dataset = tf.data.Dataset.from_tensor_slices(
                tensors=[self.live_files, self.spoof_files]
            )

            # Creates balanced batches of live and spoof data
            dataset = dataset.interleave(
                map_func=_interleave_map_fn,
                cycle_length=2,
                block_length=int(self.batch_size / 2)
            )
            dataset = dataset.batch(self.batch_size)
            return dataset


class ValidationPipeline(Pipeline):
    _DEFAULT_FEATURE_TRANSFORMS = {
        'image': Pipeline._create_preprocess_image_fn(distort=False),
        'lbp': tf.to_float,
        'daisy': tf.to_float,
        'sid': tf.to_float
    }

    def __init__(self,
                 files,
                 batch_size,
                 threads=12,
                 feature_parsers=None,
                 feature_transforms=None,
                 label_parsers=None,
                 label_transforms=None,
                 scaler_file=None):

        self.files = files
        self.batch_size = batch_size
        self.threads = threads
        self.feature_parsers = feature_parsers
        self.label_parsers = label_parsers
        self.label_transforms = label_transforms

        if feature_transforms:
            self.feature_transforms = feature_transforms
        elif scaler_file:
            scalers = np.load(scaler_file).item()
            self.feature_transforms = {
                'image': Pipeline._create_preprocess_image_fn(distort=False),
                'lbp': lambda x: tf.divide(tf.subtract(tf.to_float(x), scalers['lbp'].mean_),
                                           scalers['lbp'].scale_),
                'daisy': lambda x: tf.divide(tf.subtract(tf.to_float(x), scalers['daisy'].mean_),
                                             scalers['daisy'].scale_),
                'sid': lambda x: tf.divide(tf.subtract(tf.to_float(x), scalers['sid'].mean_),
                                           scalers['sid'].scale_),
                'hog': lambda x: tf.divide(tf.subtract(x, scalers['hog'].mean_),
                                           scalers['hog'].scale_),
                'bsif': lambda x: tf.divide(tf.subtract(x, scalers['bsif'].mean_),
                                            scalers['bsif'].scale_),
                'coa_lbp': lambda x: tf.divide(tf.subtract(x, scalers['coa_lbp'].mean_),
                                               scalers['coa_lbp'].scale_)
            }
        else:
            self.feature_transforms = self._DEFAULT_FEATURE_TRANSFORMS

    def build(self):
        with tf.name_scope('validation_data'):
            # TODO: Refactor into parent
            preprocess_fn = self._create_preprocess_fn(
                feature_parsers=(self.feature_parsers or
                                 self._DEFAULT_FEATURE_PARSERS),
                feature_transforms=self.feature_transforms,
                label_parsers=(self.label_parsers or
                               self._DEFAULT_LABEL_PARSERS),
                label_transforms=(self.label_transforms or
                                  self._DEFAULT_LABEL_TRANSFORMS)
            )
            dataset = tf.data.TFRecordDataset(self.files)
            dataset = dataset.map(
                map_func=preprocess_fn,
                num_parallel_calls=self.threads
            )
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.padded_batch(
                batch_size=1,
                padded_shapes=self._get_padded_shapes(dataset.output_shapes,
                                                      self.batch_size),
                padding_values=self._get_padded_types(dataset.output_types)
            )
            dataset = dataset.apply(tf.contrib.data.unbatch())
            return dataset

    @staticmethod
    def _get_padded_shapes(output_shapes, batch_size):
        features = output_shapes[0]
        labels = output_shapes[1]

        feature_shapes = dict()
        label_shapes = dict()

        for feature, shape in features.items():
            feature_dims = shape.dims[1:]
            feature_shapes[feature] = tf.TensorShape(
                [tf.Dimension(batch_size)] + feature_dims)

        for label, shape in labels.items():
            label_dims = shape.dims[1:]
            label_shapes[label] = tf.TensorShape(
                [tf.Dimension(batch_size)] + label_dims)
        return feature_shapes, label_shapes

    @staticmethod
    def _get_padded_types(output_types):
        features = output_types[0]
        labels = output_types[1]

        feature_values = dict()
        label_values = dict()

        for feature, dtype in features.items():
            feature_values[feature] = tf.constant(-1, dtype)

        for label, dtype in labels.items():
            label_values[label] = tf.constant(-1, dtype)
        return feature_values, label_values
