import tensorflow as tf
import sys
import os
import glob
import cv2
import numpy as np

class RecordWriter:
    def __init__(self, out_dir, in_dirs):
        self.indirs = in_dirs
        self.out_dir = out_dir

        self.img_triples = [triple for dir in in_dirs for triple in self.get_img_triples(dir)]
        # self.img_triples = self.img_triples[:100]
        self.write_tfrecords(self.img_triples, out_dir)

    def _int64_feature(self, value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def get_img_triples(self, dir_name):
        final_dir_name = os.path.join(dir_name, 'JPEGImages', '480p')
        dir_names = os.listdir(final_dir_name)
        dir_names = [os.path.join(final_dir_name, cur_dir) for cur_dir in dir_names]

        jpg_listss = [sorted(glob.glob(os.path.join(dir_name, '*.jpg'))) for dir_name in dir_names]

        res_img_triples = []
        for i, jpg_lists in enumerate(jpg_listss):
            for j, jpg_path in enumerate(jpg_lists):
                if j == 0 or j == len(jpg_lists) - 1: continue
                res_img_triples.append([jpg_lists[j-1], jpg_lists[j], jpg_lists[j+1]])
        return res_img_triples

    def center_crop(self, img):

        img_w = img.shape[1]
        img = img[:, img_w//2 - 854//2: img_w//2 + 854//2, :]
        return img

    def write_tfrecords(self, img_triples, out_dirs, mode='train'):
        train_filename = os.path.join(out_dirs, '{}.tfrecords'.format(mode))

        if not os.path.exists(out_dirs):
            os.makedirs(out_dirs)

        writer = tf.python_io.TFRecordWriter(train_filename)
        for i, img_triple in enumerate(img_triples):
            if i % 50 == 0:
                print('dealing {}'.format(i))
                print(img_triple)

            first_img = cv2.imread(img_triple[0])[:, :, ::-1].astype(np.float32)
            mid_img = cv2.imread(img_triple[1])[:, :,::-1].astype(np.float32)
            end_img = cv2.imread(img_triple[2])[:, :, ::-1].astype(np.float32)

            first_img, mid_img, end_img = list(map(lambda x: self.center_crop(x), [first_img, mid_img, end_img]))

            feature = {'first_img': self._bytes_feature(tf.compat.as_bytes(first_img.tostring())),
                       'mid_img': self._bytes_feature(tf.compat.as_bytes(mid_img.tostring())),
                       'end_img': self._bytes_feature(tf.compat.as_bytes(end_img.tostring()))}
            #            'img_shape': self._bytes_feature(tf.compat.as_bytes(img_shape.tostring()))}
            # feature = {'first_img': self._bytes_feature(tf.compat.as_bytes(img_triple[0])),
            #            'mid_img': self._bytes_feature(tf.compat.as_bytes(img_triple[1])),
            #            'end_img': self._bytes_feature(tf.compat.as_bytes(img_triple[2]))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    dir_name = '/xxx/Downloads/DAVIS'
    dir_name1 = '/xxx/Downloads/DAVIS1'
    dir_name2 = '/xxx/Downloads/DAVIS2'
    dir_names = [dir_name, dir_name1, dir_name2]
    dir_names = ['/xxx/Davis/DAVIS',
                 '/xxx/Davis/DAVIS_test_challenge',
                 '/xxx/Davis/DAVIS_trainval']
    writer = RecordWriter(out_dir='../Davis_dataset/', in_dirs=dir_names)
    pass

