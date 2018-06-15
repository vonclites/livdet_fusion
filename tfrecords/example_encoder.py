import tensorflow as tf


class ImageEncoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._image = tf.placeholder(tf.uint8)
        self._encode = tf.image.encode_jpeg(self._image,
                                            format='rgb')

    def encode(self, image):
        return self._sess.run(self._encode,
                              feed_dict={self._image: image})


def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Returns a TF-Feature of floats.
    Args:
      values: A scalar or list of values.
    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_features(encoded_image, label, image_id, height, width,
                   image_type, contacts, dataset,
                   lbp=None, coa_lbp=None, bsif=None, hog=None, daisy=None, sid=None):
    features = {
        'image': bytes_feature(encoded_image),
        'label': int64_feature(label),
        'id': int64_feature(image_id),
        'height': int64_feature(height),
        'width': int64_feature(width),
        'contacts': int64_feature(contacts),
        'type': int64_feature(image_type),
        'dataset': int64_feature(dataset)
    }
    if lbp is not None:
        features['lbp'] = tf.train.Feature(int64_list=tf.train.Int64List(value=lbp))
    if coa_lbp is not None:
        features['coa_lbp'] = tf.train.Feature(float_list=tf.train.FloatList(value=coa_lbp))
    if bsif is not None:
        features['bsif'] = tf.train.Feature(float_list=tf.train.FloatList(value=bsif))
    if hog is not None:
        features['hog'] = tf.train.Feature(float_list=tf.train.FloatList(value=hog))
    if daisy is not None:
        features['daisy'] = tf.train.Feature(int64_list=tf.train.Int64List(value=daisy))
    if sid is not None:
        features['sid'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sid))
    return features


def create_example(encoded_image, label, image_id, height, width,
                   image_type, contacts, dataset,
                   lbp=None, coa_lbp=None, bsif=None, hog=None, daisy=None, sid=None):
    features = image_features(
        encoded_image=encoded_image,
        label=label,
        image_id=image_id,
        height=height,
        width=width,
        contacts=contacts,
        image_type=image_type,
        dataset=dataset,
        lbp=lbp,
        coa_lbp=coa_lbp,
        bsif=bsif,
        hog=hog,
        daisy=daisy,
        sid=sid
    )
    return tf.train.Example(features=tf.train.Features(feature=features))
