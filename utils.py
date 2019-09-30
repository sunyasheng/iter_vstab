import os
import sys
import glob
import tensorflow as tf
import cv2
import numpy as np
from conf_tab import config
import cvbase as cvb

def write_flows(flow_lists, index, debug_out_dir):
    if not os.path.exists(debug_out_dir):
        os.makedirs(debug_out_dir)
    for key, flow in flow_lists.items():
        flow_img = cvb.flow2rgb(flow)
        # import pdb; pdb.set_trace();
        fn = os.path.join(debug_out_dir, '{}_{}.png'.format(index, key))
        cv2.imwrite(fn, np.array(flow_img*255.0).astype(np.uint8))

def write_imgs(img_lists, index, debug_out_dir):
    if not os.path.exists(debug_out_dir):
        os.makedirs(debug_out_dir)
    for i, img in img_lists.items():
        fn = os.path.join(debug_out_dir, '{}_{}.png'.format(index, i))
        cv2.imwrite(fn, img)

def linear_lr(lr_init, decay_ratio, cur_epoches):
    res_lr = lr_init - decay_ratio*cur_epoches
    return res_lr

def get_rand_H(batch_size, rand_H_min=config.rand_H_min, rand_H_max=config.rand_H_max):
    # H = tf.random_uniform([batch_size], minval=rand_H_min[0, 0], maxval=rand_H_max[0, 0], dtype=tf.float32)
    H_lists = []
    for i in range(3):
        for j in range(3):
            if (i == 2 and j == 2):
                continue
            H_lists.append(tf.random_uniform([batch_size],
                                             minval=rand_H_min[i, j],
                                             maxval=rand_H_max[i, j],
                                             dtype=tf.float32))
    H = tf.stack(H_lists, axis=1)
    return H

def get_vid_info(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Set up output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    return [n_frames, w, h, fps]

def vid2img_lists(video_path):
    [n_frames, w, h, fps] = get_vid_info(video_path)
    cap = cv2.VideoCapture(video_path)
    img_lists = []
    for i in range(n_frames):
        success, cur = cap.read()
        if not success:
            print('read failed!')
            exit()
        img_lists.append(cur)
    img_lists = [img[:, :, ::-1] for img in img_lists]
    return img_lists

def save_img_lists(img_lists, file_dir, flip_ch=True):
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    for i, img in enumerate(img_lists):
        if flip_ch:img = img[:, :, ::-1]
        cv2.imwrite(os.path.join(file_dir, '{:05d}.png'.format(i)), img)
    print('save imgs to {}'.format(file_dir))

def save2vid(img_lists, out_dir, in_fn, flip_ch=True):
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    [n_frames, w, h, fps] = get_vid_info(in_fn)
    base_name = in_fn.split('/')[-1]
    out_path = os.path.join(out_dir, base_name)
    videoWriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in range(len(img_lists)):
        img = img_lists[i][:, :, :3].astype(np.uint8)
        if flip_ch: img = img[:, :, ::-1]
        videoWriter.write(img)
    videoWriter.release()


def video2triplets(data_root, data_list):
    with open(data_list, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(' ')[0] for line in lines]
    lines = [data_root+line for line in lines]

    first_paths, mid_paths, end_paths = [], [], []
    for i in range(1, len(lines) -1):
        last_items = lines[i-1].split('/')
        items = lines[i].split('/')
        next_items = lines[i+1].split('/')
        if items[-2] != next_items[-2] or items[-2] != last_items[-2]:continue
        first_paths.append(lines[i-1])
        mid_paths.append(lines[i])
        end_paths.append(lines[i+1])

    return first_paths, mid_paths, end_paths

def optimistic_restore(session, save_file):
    """
    restore only those variable that exists in the model
    :param session:
    :param save_file:
    :return:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
    # import pdb; pdb.set_trace();
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],tf.global_variables()),tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                print("going to restore.var_name:",var_name,";saved_var_name:",saved_var_name)
                restore_vars.append(curr_var)
            else:
                print("variable not trained.var_name:",var_name)
    saver = tf.train.Saver(restore_vars)
    #
    # tmp = [var for var in tf.global_variables() if len(var.name.split(':')[0].split('/')) > 2]
    # tmp = [var for var in tmp if var.name.split(':')[0].split('/')[2] == 'pwcnet']
    # our_pwc_names_vars = dict(zip([var.name.split(':')[0] for var in tmp], tmp))

    # restore_dict = {}
    #
    # saver = tf.train.Saver(restore_dict)
    saver.restore(session, save_file)

def get_config(config):
    config_str = 'bn:{},opt:{},bs:{},ims:{},lr:{},gt:{},ks:{}'.format(
                                                  config.TRAIN.batch_norm,
                                                  config.TRAIN.optim_type,
                                                  config.TRAIN.per_gpu_batch_size,
                                                  config.TRAIN.image_input_size,
                                                  config.TRAIN.lr_init,
                                                  config.TRAIN.gated,
                                                  config.TRAIN.kernel_size)
    return config_str

def pad_img(img, pyr_lvls):
    ##img shape: [h, w, c]
    _, pad_h = divmod(img.shape[0], 2**pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(img.shape[1], 2**pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    if pad_h != 0 or pad_w != 0:
        padding = [(0, pad_h), (0, pad_w), (0, 0)]
        img = np.pad(img, padding, mode='constant', constant_values=0.)
    return img

def unpad_img(img, raw_shape):
    return img[:raw_shape[0], :raw_shape[1], :]

def resize_img(img, pyr_lvls):
    _, pad_h = divmod(img.shape[0], 2 ** pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(img.shape[1], 2 ** pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    if pad_h != 0 or pad_w != 0:
        resize_h = img.shape[0] + pad_h
        resize_w = img.shape[1] + pad_w
        img = cv2.resize(img, dsize=(resize_w, resize_h))
    return img

def back_resize_img(img, raw_shape):
    return cv2.resize(img, dsize=(raw_shape[1], raw_shape[0]))

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # compute average gradient for every variable
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    return average_grads

def get_variables_with_name(name=None, exclude_name=None, train_only=True, printable=False):
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0+
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()

    if name is None:
        d_vars = [var for var in t_vars]
    else:
        d_vars = [var for var in t_vars if name in var.name]

    if exclude_name is None:
        d_vars = [var for var in d_vars]
    else:
        d_vars = [var for var in d_vars if exclude_name not in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars

def params_count(name=None, train_only=False):
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0+
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()

    total_params = 0

    for variable in t_vars:
        if not name in variable.name: continue
        shape = variable.get_shape()
        variable_paramters = 1
        for dim in shape:
            variable_paramters *= dim.value
        total_params += variable_paramters
    param_type = 'trainable' if train_only else 'total'
    print('{} include {} {} params.'.format(name, total_params, param_type))

def file2lists(fn):
    res = []
    with open(fn, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            res.append(line.strip())
    return res

if __name__ == '__main__':
    # from conf_tab import config
    # video2triplets(config.data_root, config.TRAIN.data_path)
    pass
