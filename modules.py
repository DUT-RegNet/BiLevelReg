import tensorflow as tf
import keras.layers as KL
from keras.initializers import RandomNormal
import keras.initializers
from keras.layers import concatenate


def sample(mu, log_sigma):
    """
    sample from a normal distribution
    """
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z


class Pro_FeatureLearning(object):
    def __init__(self, num_levels=2, name='f_learning_pro'):
        self.num_levels = num_levels
        self.filters = [16, 32, 64]
        self.name = name

    def __call__(self, images, reuse=True):
        """
        Args:
        - images (batch, h, w, c, 1): input images

        Returns:
        - features_pyramid (batch, h, w, c, nch) for each scale levels:
          extracted feature pyramid (deep -> shallow order)
        """
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            features_pyramid = []
            features_params_pyramid = []
            tmp = images
            ndims = 3
            Conv = getattr(KL, 'Conv%dD' % ndims)

            for l in range(self.num_levels):
                x = tf.layers.Conv3D(self.filters[l], (3, 3, 3), (2, 2, 2), 'same', kernel_initializer='he_normal')(tmp)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv3D(self.filters[l], (3, 3, 3), (1, 1, 1), 'same', kernel_initializer='he_normal')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv3D(self.filters[l], (3, 3, 3), (1, 1, 1), 'same', kernel_initializer='he_normal')(x)
                tmp = tf.nn.leaky_relu(x, 0.1)

                features_mean = Conv(ndims, kernel_size=3, padding='same',
                                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(tmp)
                # we're going to initialize the velocity variance very low, to start stable.
                features_log_sigma = Conv(ndims, kernel_size=3, padding='same',
                                      kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                                      bias_initializer=keras.initializers.Constant(value=-10),
                                      name='log_sigma')(tmp)
                features = sample(features_mean, features_log_sigma)
                features_pyramid.append(features)

                features_params = concatenate([features_mean, features_log_sigma])
                features_params_pyramid.append(features_params)

            # return feature pyramid by ascent order
            return features_pyramid[::-1], features_params_pyramid[::-1]


class Estimator_1(object):
    def __init__(self, name='estimator_1'):
        self.filters = [48, 32, 16]
        self.name = name

    def __call__(self, features_0, features_1=None, flows_up_prev=None):
        """
        Args:
        - features_0 (batch, h, w, c, nch): feature map of image_0
        - features_0 (batch, h, w, c, nch): feature map at image_1
        - flows_up_prev (batch, h, w, c, 3): upscaled flow passed from previous scale

        Returns:
        - flows (batch, h, w, c, 3): flow
        """
        with tf.variable_scope(self.name) as vs:
            features = features_0
            for f in [features_1, flows_up_prev]:
            # for f in [features_1]: ## new
                if f is not None:
                    features = tf.concat([features, f], axis=4)

            for f in self.filters:
                conv = tf.layers.Conv3D(f, (3, 3, 3), (1, 1, 1), 'same', kernel_initializer='he_normal')(features)
                conv = tf.nn.leaky_relu(conv, 0.1)
                features = conv

            flows = tf.layers.Conv3D(3, (3, 3, 3), (1, 1, 1), 'same', kernel_initializer='he_normal')(features)
            if flows_up_prev is not None:
                # Residual connection
                flows += flows_up_prev

            return flows


class conv_block(object):
    def __init__(self, name='conv_block'):
        self.name = name

    def __call__(self, x_in, nf, strides=1):
        conv = tf.layers.Conv3D(nf, (3, 3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x_in)
        x_out = tf.nn.leaky_relu(conv, 0.2)
        return x_out



