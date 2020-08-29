import tensorflow as tf, tensorflow.keras.backend as K
import tensorflow_probability as tfp
from transformations import *

class AugMix:
    def __init__(self, means=[0, 0, 0], stds=[1, 1, 1]):
        self.means = tf.constant(means)
        self.stds = tf.constant(stds)

    def normalize(self, image):
        image = (image-self.means)/self.stds
        return tf.clip_by_value(image, 0, 1)
        
    def apply_op(self, image, level, which): 
        augmented = image
        augmented = tf.cond(which == tf.constant([0], dtype=tf.int32), lambda: rotate(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([1], dtype=tf.int32), lambda: translate_x(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([2], dtype=tf.int32), lambda: translate_y(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([3], dtype=tf.int32), lambda: shear_x(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([4], dtype=tf.int32), lambda: shear_y(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([5], dtype=tf.int32), lambda: solarize_add(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([6], dtype=tf.int32), lambda: solarize(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([7], dtype=tf.int32), lambda: posterize(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([8], dtype=tf.int32), lambda: autocontrast(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([9], dtype=tf.int32), lambda: equalize(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([10], dtype=tf.int32), lambda: color(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([11], dtype=tf.int32), lambda: contrast(image, level), lambda: augmented)
        augmented = tf.cond(which == tf.constant([12], dtype=tf.int32), lambda: brightness(image, level), lambda: augmented)
        return augmented

    def process(self, image, severity=3, width=3, depth=-1):
        """ 
        Performs AugMix data augmentation on given image.

        Parameters: 
        image (tf tensor): an image tensor with shape (x, x, 3) and values scaled to range [0, 1]
        severity (int): level of a strength of transformations (integer from 1 to 10)
        width (int): number of different chains of transformations to be mixed
        depth (int): number of transformations in one chain, -1 means random from 1 to 3
    
        Returns: 
        tensor: augmented image
    
        """
        
        alpha = 1.
        dir_dist = tfp.distributions.Dirichlet([alpha]*width)
        ws = tf.cast(dir_dist.sample(), tf.float32)
        beta_dist = tfp.distributions.Beta(alpha, alpha)
        m = tf.cast(beta_dist.sample(), tf.float32)

        mix = tf.zeros_like(image, dtype='float32')

        def outer_loop_cond(i, depth, mix):
            return tf.less(i, width)

        def outer_loop_body(i, depth, mix):
            image_aug = tf.identity(image)
            depth = tf.cond(tf.greater(depth, 0), lambda: depth, lambda: tf.random.uniform(shape=[], minval=1, maxval=3, dtype=tf.int32))

            def inner_loop_cond(j, image_aug):
                return tf.less(j, depth)

            def inner_loop_body(j, image_aug):
                which = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
                image_aug = self.apply_op(image_aug, severity, which)
                j = tf.add(j, 1)
                return j, image_aug
            
            j = tf.constant([0], dtype=tf.int32)
            j, image_aug = tf.while_loop(inner_loop_cond, inner_loop_body, [j, image_aug])

            wsi = tf.gather(ws, i)
            mix = tf.add(mix, wsi*self.normalize(image_aug))
            i = tf.add(i, 1)
            return i, depth, mix

        i = tf.constant([0], dtype=tf.int32)
        i, depth, mix = tf.while_loop(outer_loop_cond, outer_loop_body, [i, depth, mix])
        
        mixed = tf.math.scalar_mul((1 - m), self.normalize(image)) + tf.math.scalar_mul(m, mix)
        return tf.clip_by_value(mixed, 0, 1)
    
