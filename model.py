from my_pwc_net import nn
import tensorflow as tf
import numpy as np
from conf_tab import config

is_gated = config.TRAIN.gated
is_batch_norm = config.TRAIN.batch_norm
# not sure whether or not is this reason, just try
def residual_block(input_layer, output_channel, first_block=False, training=True):
    input_channel = input_layer.get_shape().as_list()[-1]

    assert input_channel == output_channel, print('Output and input channel does not match in residual blocks!!!')

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            conv1 = conv2d_padding_same(input_layer, output_channel, kernel_size=1)
        else:
            conv1 = tf.nn.relu(input_layer)
            if is_batch_norm:
                conv1 = batchnorm(conv1, training)
            conv1 = conv2d_padding_same(conv1, output_channel, kernel_size=1)

    with tf.variable_scope('conv2_in_block'):
        if is_batch_norm:
            conv2 = batchnorm(conv1, training)
        else:
            conv2 = conv1
        conv2 = tf.nn.relu(conv2)
        conv2 = conv2d_padding_same(conv2, output_channel, kernel_size=1)

    output = conv2 + input_layer
    return output

def resnet(input_tensor_batch, n, reuse=False, training=False):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    with tf.variable_scope('resnet', reuse=reuse):
        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = conv2d_padding_same(input_tensor_batch, 32, kernel_size=1)
            if is_batch_norm:
                conv0 = batchnorm(conv0, training)
            conv0 = tf.nn.relu(conv0)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' %i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 32, first_block=True, training=training)
                else:
                    conv1 = residual_block(layers[-1], 32, training=training)
                layers.append(conv1)

        conv_final = conv2d_padding_same(layers[-1], 3, kernel_size=1)
        layers.append(conv_final)
    return layers[-1]

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
    h = conv2d_padding_same(h, out_size, kernel_size=1, trainable=trainable, activate=None)
    return h

def conv_conv_pool(input_,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(config.TRAIN.reg),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_concat(inputA, input_B, n_filter, flags, name, training=True, activate=tf.nn.relu):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, flags, name, training=training, activate=activate)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, flags, name, training=True, activate=tf.nn.relu):
    """Up Convolution `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """
    resized_tensor = tf.image.resize_bilinear(tensor, size=(tf.shape(tensor)[1]*2, tf.shape(tensor)[2]*2))
    h = conv2d_padding_same(resized_tensor, n_filter, activate=None)
    h = batchnorm(h, training=training)
    h = activate(h)
    return h
    # return tf.layers.conv2d_transpose(
    #     tensor,
    #     filters=n_filter,
    #     kernel_size=2,
    #     strides=2,
    #     kernel_regularizer=tf.contrib.layers.l2_regularizer(config.TRAIN.reg),
    #     name="upsample_{}".format(name))


def make_unet(X, training=True, out_size = 3,  flags=None):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = X
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, flags, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, flags, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, flags, name=6, training=training)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, flags, name=7, training=training)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, flags, name=8, training=training)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8, flags, name=9, training=training)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        out_size, (1, 1),
        name='final',
        activation=None,
        padding='same')


def u_net(inputs, training=True, trainable=True, out_size=2):
    h = inputs
    filter_nums = [32, 32, 32, 32]
    # with tf.variable_scope('flow_net_{}'.format(name)):
    mid_feat = []
    for k, n_dim in enumerate(filter_nums):
        h = batchnorm(h, training)
        h = conv2d_padding_same(h, n_dim, activate=tf.nn.relu, trainable=trainable)
        if k != len(filter_nums) - 1:
            mid_feat.append(h)
            h = maxpool2d_same(h)

    for n_dim, pre_f in zip(filter_nums[:-1][::-1], mid_feat[::-1]):
        shape_f = tf.shape(pre_f)
        h = tf.image.resize_bilinear(h, (shape_f[1], shape_f[2]))
        h = tf.concat([h, pre_f], axis=-1)
        h = batchnorm(h, training)
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
    flow_pred0, _ = nn(x_tnsr0, reuse=False)
    flow_pred0 = flow_pred0[:, :, :, ::-1]


    x_tnsr2 = tf.stack([img_end, img_s], axis=1)
    flow_pred2, _ = nn(x_tnsr2, reuse=True)
    flow_pred2 = flow_pred2[:, :, :, ::-1]

    with tf.variable_scope('stabnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('first2mid_warp', reuse=tf.AUTO_REUSE):
            warped_first = flow_warp(img_first, flow_pred0)
        with tf.variable_scope('end2mid_warp', reuse=tf.AUTO_REUSE):
            warped_end = flow_warp(img_end, flow_pred2)
        with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
            # mask = u_net(tf.concat([warped_first, warped_end], axis=-1), training=training, out_size=1)
            # mask = (mask + 1.0) * 0.5
            # img_int = warped_first * mask + warped_end * (1.0 - mask)
            # img_int = tf.clip_by_value(img_int, 0, 1)
            img_int = u_net(tf.concat([warped_first, warped_end], axis=-1), training=training, out_size= 3)

    x_tnsr1 = tf.stack([img_mid, img_int], axis=1)
    flow_pred1, _ = nn(x_tnsr1, reuse=True)
    flow_pred1 = flow_pred1[:, :, :, ::-1]

    with tf.variable_scope('stabnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mid2int', reuse=tf.AUTO_REUSE):
            warped_mid = flow_warp(img_mid, flow_pred1)
        with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
            # img_out = resnet_1x1(tf.concat([warped_mid, img_int], axis=-1),
            #                                  training=training, trainable=trainable,
            #                                  reuse=reuse, out_size=3)
            img_out = resnet(tf.concat([warped_mid,  img_int], axis=-1), 5, reuse=reuse, training=training)

            # mask1 = (mask1 + 1.0) * 0.5
            # img_out = warped_mid * mask1 + img_int * (1 - mask1)
    return img_int, img_out, [warped_first, warped_end, warped_mid]

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
            img_int = make_unet(tf.concat([warped_first, warped_end], axis=-1), training=training, out_size=3)

    x_tnsr1 = tf.stack([mid_img, img_int], axis=1)
    flow_pred1, _ = nn(x_tnsr1, reuse=tf.AUTO_REUSE)
    flow_pred1 = flow_pred1[:, :, :, ::-1]

    with tf.variable_scope('stabnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mid2int', reuse=tf.AUTO_REUSE):
            warped_mid = flow_warp(mid_img, flow_pred1)
        with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
            # img_out = resnet_1x1(tf.concat([warped_mid, img_int], axis=-1),
            #                      training=training, trainable=trainable, reuse=reuse)
            img_out = resnet(tf.concat([warped_mid,  img_int], axis=-1), 5, reuse=reuse, training=training)
    debug_out = [warped_first, warped_end, img_int, flow_pred0, flow_pred1, flow_pred2]
    return img_out, debug_out

def test_training_model():
    img_first, img_s, img_end, img_mid = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),\
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),\
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),\
                                         tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    img_int, img_out, [warped_first, warped_end, warped_mid] = training_stab_model(img_first, img_s, img_end, img_mid) #img_first, img_s, img_end, img_mid,

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

def test_flow_warp(img_first, img_s):
    x_tnsr0 = tf.stack([img_first, img_s], axis=1)
    flow_pred0, _ = nn(x_tnsr0, reuse=tf.AUTO_REUSE)
    flow_pred0 = flow_pred0[:, :, :, ::-1]
    warped = flow_warp(img_first, flow_pred0)
    return warped

if __name__ == '__main__':
    import os
    import cv2
    from utils import optimistic_restore
    from pwc_tab import pwc_opt
    out_dir = './tmp_out'
    batch_idx = 10

    s_img_path = os.path.join(out_dir, '{}_s.png'.format(batch_idx))
    first_img_path = os.path.join(out_dir, '{}_first.png'.format(batch_idx))
    end_img_path = os.path.join(out_dir, '{}_end.png'.format(batch_idx))
    mid_img_path = os.path.join(out_dir, '{}_mid.png'.format(batch_idx))

    img_first = cv2.imread(first_img_path).astype(np.float32)[:,:,::-1]
    img_s = cv2.imread(s_img_path).astype(np.float32)[:,:,::-1]
    img_end = cv2.imread(end_img_path).astype(np.float32)[:, :, ::-1]
    img_mid = cv2.imread(mid_img_path).astype(np.float32)[:, :, ::-1]

    img_first = img_first / 255.
    img_s = img_s /255.
    img_end = img_end / 255.
    img_mid = img_mid /255.


    img_first = np.expand_dims(img_first, 0)
    img_s = np.expand_dims(img_s, 0)
    img_mid = np.expand_dims(img_mid, 0)
    img_end = np.expand_dims(img_end, 0)

    # warped = test_flow_warp(img_first, img_s)
    img_int, img_out, [warped_first, warped_end, warped_mid] = training_stab_model(img_first,
                                                                                   img_s, img_end, img_mid, training=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    sess.run(tf.global_variables_initializer())

    optimistic_restore(sess, pwc_opt.ckpt_path) #必须放在下面，否则会被覆盖


    [warped_first, warped_end] = sess.run([warped_first, warped_end])
    # import pdb; pdb.set_trace();
    warped_first = warped_first[0][:,:,::-1]
    warped_end = warped_end[0][:,:,::-1]

    # warped_np = np.clip(warped_np, 0, 1.)
    cv2.imwrite('warped_first.png', np.array(warped_first*255).astype(np.uint8))
    cv2.imwrite('warped_end.png', np.array(warped_end*255).astype(np.uint8))
    # test_training_model()
    # test_testing_model()
    # input_tensor_batch = tf.random_uniform(shape=[2, 16, 16, 3])
    # # out = resnet(input_tensor_batch, 5)
    # out = make_unet(input_tensor_batch)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(out).shape)