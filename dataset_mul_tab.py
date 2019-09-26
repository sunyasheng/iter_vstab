import tensorflow as tf
from conf_tab import config
from utils import get_rand_H

image_input_size = config.TRAIN.image_input_size

def mkdataset(first_names, mid_names, end_names, batch_size=32,
              gpu_ind=0, num_gpu=1, num_parallel=24):
    def _parse_function(first_name, mid_name, end_name):
        first_img_str = tf.read_file(first_name)
        first_img_decoded = tf.image.decode_jpeg(first_img_str, channels=3)
        first_img_decoded.set_shape([None, None, 3])

        mid_img_str = tf.read_file(mid_name)
        mid_img_decoded = tf.image.decode_jpeg(mid_img_str, channels=3)
        mid_img_decoded.set_shape([None, None, 3])

        s_img_str = tf.read_file(mid_name)
        s_img_decoded = tf.image.decode_jpeg(s_img_str, channels=3)
        s_img_decoded.set_shape([None, None, 3])
        s_img_decoded = tf.contrib.image.transform(
            s_img_decoded,
            get_rand_H(batch_size=1),
            interpolation='BILINEAR')

        end_img_str = tf.read_file(end_name)
        end_img_decoded = tf.image.decode_jpeg(end_img_str, channels=3)
        end_img_decoded.set_shape([None, None, 3])

        # return first_img_decoded, mid_img_decoded, end_img_decoded, s_img_decoded
        p = tf.random_uniform([], 0, 1)
        tmp = tf.cond(p > 0.5, lambda: tf.concat([first_img_decoded, mid_img_decoded, end_img_decoded, s_img_decoded], axis=2),
                      lambda: tf.concat([end_img_decoded, mid_img_decoded, first_img_decoded, s_img_decoded], axis=2))
        #
        crop_w = config.TRAIN.image_input_size# + config.TRAIN.image_input_size // 8
        crop_h = crop_w
        # tmp = tf.slice(tmp, [(tf.shape(tmp)[0] - crop_h)//2, (tf.shape(tmp)[1] - crop_w)//2, 0],
        #                [crop_w, crop_h, -1])
        tmp = tf.random_crop(tmp, [crop_w, crop_h, 3*4])
        #
        tmp = tf.image.random_flip_left_right(tmp)
        tmp = tf.image.random_flip_up_down(tmp)
        # # tmp = tf.image.resize_bilinear(tf.expand_dims(tmp, 0), [image_input_size, image_input_size])
        #
        tmp = tf.split(tmp, num_or_size_splits=4, axis=2)
        #
        # tmp1 = tf.concat([tmp[0], tmp[1], tmp[2]], axis=2)
        # tmp_out = tf.random_crop(tmp1, [config.TRAIN.image_input_size, config.TRAIN.image_input_size, 3*3])
        # tmp_out = tf.split(tmp_out, num_or_size_splits=3, axis=2)
        # s_img = tf.random_crop(tmp[3], [config.TRAIN.image_input_size, config.TRAIN.image_input_size, 3])
        #
        # return tmp_out[0], tmp_out[1], tmp_out[2], s_img
        return tmp[0], tmp[1], tmp[2], tmp[3]

    dataset = tf.data.Dataset.from_tensor_slices((first_names, mid_names, end_names))
    dataset = dataset.shard(num_gpu, gpu_ind)
    dataset = dataset.repeat(200).shuffle(buffer_size=72000)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(_parse_function, batch_size=batch_size, num_parallel_calls=num_parallel))
    dataset = dataset.prefetch(32)

    iterator = dataset.make_initializable_iterator()
    first_img, second_img, end_img, s_img = iterator.get_next()
    return first_img, second_img, end_img, s_img, iterator
