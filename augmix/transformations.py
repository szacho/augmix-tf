import math
import tensorflow as tf, tensorflow.keras.backend as K
from .helpers import *

def rotate(image, level):
    degrees = float_parameter(sample_level(level), 30)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    degrees = tf.cond(rand_var > 0.5, lambda: degrees, lambda: -degrees)

    pi = tf.constant(math.pi)
    angle = pi*degrees/180 # convert degrees to radians
    angle = tf.cast(angle, tf.float32)
    # define rotation matrix
    c1 = tf.math.cos(angle)
    s1 = tf.math.sin(angle)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(image, rotation_matrix)
    return transformed

def translate_x(image, level):
    lvl = int_parameter(sample_level(level), image.shape[0] / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_x_matrix = tf.reshape(tf.concat([one,zero,zero, zero,one,lvl, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(image, translate_x_matrix)
    return transformed

def translate_y(image, level):
    lvl = int_parameter(sample_level(level), image.shape[0] / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_y_matrix = tf.reshape(tf.concat([one,zero,lvl, zero,one,zero, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(image, translate_y_matrix)
    return transformed

def shear_x(image, level):
    lvl = float_parameter(sample_level(level), 0.3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    s2 = tf.math.sin(lvl)
    shear_x_matrix = tf.reshape(tf.concat([one,s2,zero, zero,one,zero, zero,zero,one],axis=0), [3,3])   

    transformed = affine_transform(image, shear_x_matrix)
    return transformed

def shear_y(image, level):
    lvl = float_parameter(sample_level(level), 0.3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    c2 = tf.math.cos(lvl)
    shear_y_matrix = tf.reshape(tf.concat([one,zero,zero, zero,c2,zero, zero,zero,one],axis=0), [3,3])   
    
    transformed = affine_transform(image, shear_y_matrix)
    return transformed

def solarize(image, level):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    threshold = float_parameter(sample_level(level), 1)
    return tf.where(image < threshold, image, 1 - image)

def solarize_add(image, level):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    threshold = float_parameter(sample_level(level), 1)
    addition = float_parameter(sample_level(level), 0.5)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    addition = tf.cond(rand_var > 0.5, lambda: addition, lambda: -addition)

    added_image = tf.cast(image, tf.float32) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 1), tf.float32)
    return tf.where(image < threshold, added_image, image)

def posterize(image, level):
    lvl = int_parameter(sample_level(level), 8)
    shift = 8 - lvl
    shift = tf.cast(shift, tf.uint8)
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)
    image = tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
    return tf.cast(tf.clip_by_value(tf.math.divide(image, 255), 0, 1), tf.float32)

def autocontrast(image, _):
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)

    def scale_channel(image):
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return tf.cast(tf.clip_by_value(tf.math.divide(image, 255), 0, 1), tf.float32)

def equalize(image, _):
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)

    def scale_channel(im, c):
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                        lambda: im,
                        lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)

    return tf.cast(tf.clip_by_value(tf.math.divide(image, 255), 0, 1), tf.float32)

def color(image, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    blended = blend(degenerate, image, factor)
    return tf.cast(tf.clip_by_value(tf.math.divide(blended, 255), 0, 1), tf.float32)

def brightness(image, level):
    delta = float_parameter(sample_level(level), 0.5) + 0.1
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    delta = tf.cond(rand_var > 0.5, lambda: delta, lambda: -delta) 
    return tf.image.adjust_brightness(image, delta=delta)

def contrast(image, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    factor = tf.reshape(factor, [])
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    factor = tf.cond(rand_var > 0.5, lambda: factor, lambda: 1.9 - factor  )

    return tf.image.adjust_contrast(image, factor)


