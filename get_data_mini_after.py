import tensorflow as tf
import sys
import os
import glob
import numpy as np
import cv2
from utils import get_rand_H
from conf_tab import config

class RecordReader:
    def __init__(self, in_dir):
        self.in_dir = in_dir
        # self.file_list = glob.glob(os.path.join(self.in_dir, '*.tfrecords'))

        self.train_record = os.path.join(in_dir, 'train.tfrecords')
        assert os.path.exists(self.train_record), print('train tfrecords does not exists')

    def get_img(self, img_path):
        img_str = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_str, channels=3)
        img_decoded.set_shape([None, None, 3])
        return img_decoded

    # def __len__(self):
    #     return len(self.file_list)

    def read_and_decode(self, shuffle=True):
        filename_queue = tf.train.string_input_producer([self.train_record], num_epochs=None, shuffle=shuffle)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feature = {'first_img': tf.FixedLenFeature([], tf.string),
                   'mid_img': tf.FixedLenFeature([], tf.string),
                   'end_img': tf.FixedLenFeature([], tf.string)}

        features = tf.parse_single_example(serialized_example,
                               features=feature)

        first_img = tf.decode_raw(features['first_img'], tf.float32)
        mid_img = tf.decode_raw(features['mid_img'], tf.float32)
        end_img = tf.decode_raw(features['end_img'], tf.float32)

        first_img = tf.reshape(first_img, [480, 854, 3])
        mid_img = tf.reshape(mid_img, [480, 854, 3])
        end_img = tf.reshape(end_img, [480, 854, 3])
        s_img = tf.contrib.image.transform(
            mid_img,
            get_rand_H(batch_size=1),
            interpolation='BILINEAR')

        p = tf.random_uniform([], 0, 1)
        tmp = tf.cond(p > 0.5,
                      lambda: tf.concat([first_img, mid_img, end_img, s_img], axis=2),
                      lambda: tf.concat([end_img, mid_img, first_img, s_img], axis=2))
        crop_w = config.TRAIN.image_input_size  # + config.TRAIN.image_input_size // 8
        crop_h = crop_w
        tmp = tf.random_crop(tmp, [crop_w, crop_h, 3 * 4])
        tmp = tf.image.random_flip_left_right(tmp)
        tmp = tf.image.random_flip_up_down(tmp)
        tmp = tf.split(tmp, num_or_size_splits=4, axis=2)

        return tmp[0], tmp[1], tmp[2], tmp[3]

    def run(self):
        first_img, mid_img, end_img, s_img = self.read_and_decode()
        first_img_t_batch, mid_img_t_batch, end_img_t_batch, s_img_t_batch = tf.train.shuffle_batch(
            [first_img, mid_img, end_img, s_img],
            batch_size=1, capacity=20000,
            min_after_dequeue=80, num_threads=6)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for batch_idx in range(1000):
                first_img_np, mid_img_np, end_img_np, s_img_np = sess.run([first_img_t_batch, mid_img_t_batch, end_img_t_batch, s_img_t_batch])

                print(first_img_np.shape)
                cv2.imwrite('{}_mid.png'.format(batch_idx), mid_img_np[0, :, :, ::-1])
                cv2.imwrite('{}_s.png'.format(batch_idx), s_img_np[0, :, :, ::-1])
            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    reader = RecordReader(in_dir='./Davis_dataset')
    # print(len(reader))
    reader.run()