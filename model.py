from my_pwc_net import nn
import tensorflow as tf
import numpy as np
from conf_tab import config

is_gated = config.TRAIN.gated

def gated_conv2d_padding_same(inputs, numfilter, kernel_size=3, trainable=True, activate=None):
    h = tf.layers.conv2d(inputs, 2*numfilter, kernel_size, padding='same',
                            kernel_initializer=tf.variance_scaling_initializer(),
                            trainable=trainable, activation=None)
    x1, x2 = tf.split(h, 2, axis=-1)
    if activate is not None: x1 = activate(x1)
    x2 = tf.nn.sigmoid(x2)
    return x1 * x2

def conv2d_padding_same(inputs, numfilter, kernel_size=3, trainable=True, activate=None):
    if is_gated == False:
        return tf.layers.conv2d(inputs, numfilter, kernel_size, padding='same',
                                kernel_initializer=tf.variance_scaling_initializer(),
                                trainable=trainable, activation=activate)
    else:
        return gated_conv2d_padding_same(inputs, numfilter, kernel_size=3, trainable=True, activate=None)

def batchnorm(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training)

def maxpool2d_same(inputs, poolsize=2, stride=2):
    return tf.layers.max_pooling2d(inputs, pool_size=poolsize, strides=stride, padding='same')

def res_block(inputs, training=True, trainable=True, reuse=False):
    # with tf.variable_scope('res_block', reuse=reuse):
    h = inputs
    with tf.variable_scope('batchnorm1', reuse=reuse):
        h = batchnorm(h, training=training)
        h = conv2d_padding_same(h, 32, kernel_size=1, trainable=trainable, activate=tf.nn.relu)
    with tf.variable_scope('batchnorm2', reuse=reuse):
        h = batchnorm(h, training=training)
        h = conv2d_padding_same(h, 32, kernel_size=1, trainable=trainable, activate=tf.nn.relu)
    h = h + inputs
    # h = tf.nn.relu(h)
    return h

def resnet_1x1(inputs, training=True, trainable=True, reuse=False, out_size=3):
    h = inputs
    h = batchnorm(h, training=training)
    h = conv2d_padding_same(h, 32, kernel_size=1, trainable=trainable, activate=tf.nn.relu)
    for i in range(5):
        with tf.variable_scope('res_block_{}'.format(i), reuse=reuse):
            h = res_block(h, training=training, trainable=trainable, reuse=reuse)
    h = batchnorm(h, training=training)
    h = conv2d_padding_same(h, 3, kernel_size=1, trainable=trainable, activate=None)
    return h

def u_net(inputs, training=True, trainable=True, out_size=2):
    h = inputs
    filter_nums = [64, 128, 256, 256]
    # with tf.variable_scope('flow_net_{}'.format(name)):
    mid_feat = []
    for k, n_dim in enumerate(filter_nums):
        h = batchnorm(h, training)
        h = conv2d_padding_same(h, n_dim, activate=tf.nn.relu, trainable=trainable)
        if k != len(filter_nums) - 1:
            mid_feat.append(h)
            h = maxpool2d_same(h)

    for n_dim, pre_f in zip(filter_nums[:-1][::-1], mid_feat[::-1]):
        h = batchnorm(h, training)
        shape_f = tf.shape(pre_f)
        h = tf.image.resize_bilinear(h, (shape_f[1], shape_f[2]))
        h = tf.concat([h, pre_f], axis=-1)
        h = conv2d_padding_same(h, n_dim, activate=tf.nn.relu, trainable=trainable)

    h = batchnorm(h, training)
    h = conv2d_padding_same(h, out_size, kernel_size=3, activate=None, trainable=trainable)

    return h

def flow_warp(img_from, flow_pred, reuse=False):

    warped_img = tf.contrib.image.dense_image_warp(image=img_from, flow=flow_pred)
    warped_img = tf.reshape(warped_img, tf.shape(img_from))

    return warped_img

## TODO: replace the traditional conv op with gated conv op
def training_stab_model(img_first, img_s, img_end, img_mid, reuse=False, training=True, trainable=True):
    x_tnsr0 = tf.stack([img_first, img_s], axis=1)
    flow_pred0, _ = nn(x_tnsr0, reuse=tf.AUTO_REUSE)
    flow_pred0 = flow_pred0[:, :, :, ::-1]


    x_tnsr2 = tf.stack([img_end, img_s], axis=1)
    flow_pred2, _ = nn(x_tnsr2, reuse=tf.AUTO_REUSE)
    flow_pred2 = flow_pred2[:, :, :, ::-1]

    with tf.variable_scope('stabnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('first2mid_warp', reuse=tf.AUTO_REUSE):
            warped_first = flow_warp(img_first, flow_pred0)
        with tf.variable_scope('end2mid_warp', reuse=tf.AUTO_REUSE):
            warped_end = flow_warp(img_end, flow_pred2)
        with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
            img_int = u_net(tf.concat([warped_first, warped_end], axis=-1), training=training, out_size= 3)

    x_tnsr1 = tf.stack([img_mid, img_int], axis=1)
    flow_pred1, _ = nn(x_tnsr1, reuse=tf.AUTO_REUSE)
    flow_pred1 = flow_pred1[:, :, :, ::-1]

    with tf.variable_scope('stabnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mid2int', reuse=tf.AUTO_REUSE):
            warped_mid = flow_warp(img_mid, flow_pred1)
        with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
            img_out = resnet_1x1(tf.concat([warped_mid, img_int], axis=-1),
                                             training=training, trainable=trainable, reuse=reuse)
    return img_int, img_out, [warped_first, warped_end]

def testing_stab_model(first_img, mid_img, end_img, reuse=False, training=True, trainable=True):
    x_tnsr0 = tf.stack([first_img, end_img], axis=1)
    flow_pred0, _ = nn(x_tnsr0, reuse=tf.AUTO_REUSE)
    flow_pred0 = flow_pred0[:, :, :, ::-1]

    x_tnsr2 = tf.stack([end_img, first_img], axis=1)
    flow_pred2, _ = nn(x_tnsr2, reuse=tf.AUTO_REUSE)
    flow_pred2 = flow_pred2[:, :, :, ::-1]

    with tf.variable_scope('stabnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('first2mid_warp', reuse=tf.AUTO_REUSE):
            warped_first = flow_warp(first_img, flow_pred0*0.5)
        with tf.variable_scope('end2mid_warp', reuse=tf.AUTO_REUSE):
            warped_end = flow_warp(end_img, flow_pred2*0.5)
        with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
            img_int = u_net(tf.concat([warped_first, warped_end], axis=-1), training=training, out_size=3)

    x_tnsr1 = tf.stack([mid_img, img_int], axis=1)
    flow_pred1, _ = nn(x_tnsr1, reuse=tf.AUTO_REUSE)
    flow_pred1 = flow_pred1[:, :, :, ::-1]

    with tf.variable_scope('stabnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mid2int', reuse=tf.AUTO_REUSE):
            warped_mid = flow_warp(mid_img, flow_pred1)
        with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
            img_out = resnet_1x1(tf.concat([warped_mid, img_int], axis=-1),
                                 training=training, trainable=trainable, reuse=reuse)
    debug_out = [warped_first, warped_end, img_int]
    return img_out, debug_out

def test_training_model():
    img_first, img_s, img_end, img_mid = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),\
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),\
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),\
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    img_int, img_out, [warped_first, warped_end] = training_stab_model(img_first, img_s, img_end, img_mid) #img_first, img_s, img_end, img_mid,

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    img_int_np, img_out_np = sess.run([img_int, img_out], feed_dict={img_first:np.random.uniform(size=[2, 256, 256, 3]),
                                                                    img_s: np.random.uniform(size=[2, 256, 256, 3]),
                                                                    img_end: np.random.uniform(size=[2, 256, 256, 3]),
                                                                    img_mid: np.random.uniform(size=[2, 256, 256, 3])})
    # import pdb; pdb.set_trace();
    print(img_int_np.shape, img_out_np.shape)

def test_testing_model():
    img_first, img_end, img_mid = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]), \
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]), \
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    img_out = testing_stab_model(img_first, img_mid, img_end)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    img_out_np = sess.run(img_out,
                                      feed_dict={img_first: np.random.uniform(size=[2, 256, 256, 3]),
                                                 img_end: np.random.uniform(size=[2, 256, 256, 3]),
                                                 img_mid: np.random.uniform(size=[2, 256, 256, 3])})
    import pdb; pdb.set_trace();
    print(img_out_np.shape)


if __name__ == '__main__':
    test_training_model()
    # test_testing_model()