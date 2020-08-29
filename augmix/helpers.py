import tensorflow as tf, tensorflow.keras.backend as K

def int_parameter(level, maxval):
    return tf.cast(level * maxval / 10, tf.int32)

def float_parameter(level, maxval):
    return tf.cast((level) * maxval / 10., tf.float32)

def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)
    
def affine_transform(image, transform_matrix):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    DIM = image.shape[0]
    XDIM = DIM%2 #fix for size 331
    
    x = tf.repeat(tf.range(DIM//2,-DIM//2,-1), DIM)
    y = tf.tile(tf.range(-DIM//2,DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack([x, y, z])
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(transform_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[DIM,DIM,3])

def blend(image1, image2, factor):
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)