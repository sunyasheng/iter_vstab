import dataset_mul_tab as dataset_mul
import tensorflow as tf
from conf_tab import config
from utils import video2triplets
import cv2
import os
import numpy as np

def generate_data(fn):
    first_names_p = tf.placeholder(tf.string, shape=[None])
    mid_names_p = tf.placeholder(tf.string, shape=[None])
    end_names_p = tf.placeholder(tf.string, shape=[None])

    first_names, mid_names, end_names = video2triplets(config.data_root, config.TRAIN.data_path)
    # first_names, mid_names, end_names = ['in_img.png',], ['in_img.png'], ['in_img.png']

    first_img_t, mid_img_t, end_img_t, s_img_t, iterator_gpu \
        = dataset_mul.mkdataset(first_names_p, mid_names_p, end_names_p,
                                4, gpu_ind=0,
                                num_gpu=1)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    sess.run(iterator_gpu.initializer, feed_dict={first_names_p: first_names,
                                              mid_names_p: mid_names,
                                              end_names_p: end_names})

    first_img_np, mid_img_np, end_img_np, s_img_np = sess.run(
        [first_img_t, mid_img_t, end_img_t, s_img_t],
        feed_dict={
            first_names_p: first_names,
            mid_names_p: mid_names,
            end_names_p: end_names
        })
    first_img_np, mid_img_np, end_img_np, s_img_np = list(map(lambda x: x[0], [first_img_np, mid_img_np, end_img_np, s_img_np]))

    # cv2.imwrite(fn+'_first_img.png', first_img_np[:,:,::-1])
    cv2.imwrite(fn+'_mid_img.png', mid_img_np[:,:,::-1])
    # cv2.imwrite(fn+'_end_img.png', end_img_np[:,:,::-1])
    cv2.imwrite(fn+'_s_img.png', s_img_np[:,:,::-1])
    # up_i, down_i = s_img_np.shape[0] //2 - 128, s_img_np.shape[0] //2 + 128
    # left_j, right_j = s_img_np.shape[1] //2 - 128, s_img_np.shape[1] //2 +128
    # assert np.all(s_img_np[up_i:down_i,left_j:right_j,::-1]==255), print('not 255')
    # cv2.imwrite(fn+'_s_img.png', s_img_np[up_i:down_i,left_j:right_j,::-1])

if __name__ == '__main__':
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    #
    # for i in range(10):
    #     tmp_np = sess.run(tf.random_uniform([1]))
    #     print(tmp_np)
    #     print(tmp_np.shape)
    # import numpy as np
    # in_img = np.array(np.ones(shape=(480, 854))*255.).astype(np.uint8)
    # cv2.imwrite('./in_img.png', in_img)

    fn = 'out_imgs/'
    if not os.path.exists(fn): os.makedirs(fn)
    fn = fn + '_data'
    for i in range(30):
        if i %10==0:print(i)
        generate_data(fn+str(i))