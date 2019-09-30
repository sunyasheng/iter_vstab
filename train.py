import tensorflow as tf
from conf_tab import config
import os
import dataset_mul_tab as dataset_mul
from model import training_stab_model, testing_stab_model
import numpy as np
import time
from vgg19 import Vgg19
import datetime
from tensorflow.python import debug as tf_debug
import sys
import cv2
from pwc_tab import pwc_opt
from utils import video2triplets, optimistic_restore, get_config, pad_img, unpad_img
from utils import get_variables_with_name, average_gradients, get_available_gpus, params_count
from utils import save_img_lists, linear_lr, write_imgs, write_flows
from get_data_mini_after import RecordReader
from PIL import Image
ckpt_path = '../tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

beta1 = config.TRAIN.beta1
beta2 = config.TRAIN.beta2
lr_init = config.TRAIN.lr_init
pwc_lr_init = config.TRAIN.pwc_lr_init
lr_decay = config.TRAIN.lr_decay
checkpoint_path = config.TRAIN.checkpoint_path
loss_type = config.TRAIN.loss_type
optim_type = config.TRAIN.optim_type
debug = config.DEBUG
per_gpu_batch_size = config.TRAIN.per_gpu_batch_size
image_input_size = config.TRAIN.image_input_size
pwc_decay_ratio = config.TRAIN.pwc_decay_ratio
decay_ratio = config.TRAIN.decay_ratio
pwc_freeze_epoch = config.TRAIN.pwc_freeze_epoch
lr_stable_epoch = config.TRAIN.lr_stable_epoch
pwc_lr_stable_epoch = config.TRAIN.pwc_lr_stable_epoch

timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")


def build_model(first_img_t, mid_img_t, end_img_t, s_img_t, vgg_data_dict=None, reuse_all=False):

    first_img_t = tf.cast(first_img_t, tf.float32)/255.0
    mid_img_t = tf.cast(mid_img_t, tf.float32)/255.0
    end_img_t = tf.cast(end_img_t, tf.float32)/255.0
    s_img_t = tf.cast(s_img_t, tf.float32)/255.0

    assert vgg_data_dict is not None, 'Invalid vgg data dict'
    vgg = Vgg19(vgg_data_dict)
    img_int, img_out, [warped_first, warped_end, warped_mid] = training_stab_model(first_img_t, s_img_t, end_img_t, mid_img_t)

    int_feat = vgg.relu4_4(img_int)
    out_feat = vgg.relu4_4(img_out)
    s_feat = vgg.relu4_4(s_img_t)

    vgg_int_loss = tf.reduce_mean(tf.square(s_feat - int_feat), axis=[0, 1, 2, 3])
    vgg_out_loss = tf.reduce_mean(tf.square(s_feat - out_feat), axis=[0, 1, 2, 3])

    l1_int_loss = tf.losses.absolute_difference(img_int, s_img_t)
    l1_out_loss = tf.losses.absolute_difference(img_out, s_img_t)

    summary = [tf.summary.image('first_img', first_img_t),
               tf.summary.image('end_img', end_img_t),
               tf.summary.image('mid_img', mid_img_t),
               tf.summary.image('s_img', s_img_t),
               tf.summary.image('warped_first', tf.clip_by_value(warped_first, 0, 1)),
               tf.summary.image('warped_end', tf.clip_by_value(warped_end, 0, 1)),
               tf.summary.image('warped_mid', tf.clip_by_value(warped_mid, 0, 1)),
               tf.summary.image('int_img', tf.clip_by_value(img_int, 0, 1)),
               tf.summary.image('out_img', tf.clip_by_value(img_out, 0, 1)),
               tf.summary.scalar('vgg_int_loss', vgg_int_loss),
               tf.summary.scalar('vgg_out_loss', vgg_out_loss),
               tf.summary.scalar('l1_int_loss', l1_int_loss),
               tf.summary.scalar('l1_out_loss', l1_out_loss)]
    tot_loss = vgg_int_loss + vgg_out_loss + \
                l1_int_loss + l1_out_loss
    # tot_loss = vgg_int_loss + l1_int_loss
    # tot_loss = l1_int_loss + l1_out_loss
    return tot_loss, summary


def train(args):
    global num_gpu
    global batch_size

    num_gpu = 1
    batch_size = per_gpu_batch_size*num_gpu

    lr_init = config.TRAIN.lr_init
    pwc_lr_init = config.TRAIN.pwc_lr_init

    record_reader = RecordReader(config.TRAIN.tf_records_path)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
        opt=tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2)
        pwc_lr_v = tf.Variable(pwc_lr_init, trainable=False)
        pwcnet_opt = tf.train.AdamOptimizer(pwc_lr_v, beta1=beta1, beta2=beta2)

    vgg_data_dict = np.load(config.TRAIN.vgg19_npy_path, encoding='latin1').item()

    first_img_t, mid_img_t, end_img_t, s_img_t = record_reader.read_and_decode()
    first_img_t_batch, mid_img_t_batch, end_img_t_batch, s_img_t_batch = tf.train.shuffle_batch(
        [first_img_t, mid_img_t, end_img_t, s_img_t],
        batch_size=batch_size, capacity=12000,
        min_after_dequeue=160, num_threads=4)

    reuse_all = False

    tower_grads, tower_pwc_grads = [], []
    tower_loss = []

    for d in range(0, num_gpu):
        print("dealing {}th gpu".format(d))
        with tf.device('/gpu:%s' % d):
            with tf.name_scope('%s_%s' % ('tower', d)):
                print("build model!!!")
                tot_loss_gpu, summary \
                    = build_model(first_img_t_batch, mid_img_t_batch, end_img_t_batch, s_img_t_batch, vgg_data_dict, reuse_all=reuse_all)

                if not reuse_all:
                    vars_trainable = get_variables_with_name(name='stabnet', exclude_name='pwcnet', train_only = True)
                    grads = opt.compute_gradients(tot_loss_gpu, var_list=vars_trainable)
                    pwc_vars_trainable = get_variables_with_name(name='pwcnet', exclude_name='stabnet', train_only= True)
                    pwc_grads = opt.compute_gradients(tot_loss_gpu, var_list=pwc_vars_trainable)

                for i, (g, v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_norm(g, 5), v)
                for i, (g, v) in enumerate(pwc_grads):
                    if g is not None:
                        pwc_grads[i] = (tf.clip_by_norm(g, 5), v)

                tower_grads.append(grads)
                tower_pwc_grads.append(pwc_grads)
                tower_loss.append(tot_loss_gpu)

                reuse_all = True
        if num_gpu == 1:
            with tf.device('/gpu:0'):
                mse_loss = tf.reduce_mean(tf.stack(tower_loss, 0), 0)
                mean_grads = average_gradients(tower_grads)
                mean_pwc_grads = average_gradients(tower_pwc_grads)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?stabnet')
                update_pwc_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?pwcnet')
                with tf.control_dependencies(update_ops):
                    minimize_op = opt.apply_gradients(mean_grads)
                with tf.control_dependencies(update_pwc_ops):
                    minimize_pwc_op = pwcnet_opt.apply_gradients(mean_pwc_grads)

        else:
            mse_loss = tf.reduce_mean(tf.stack(tower_loss, 0), 0)
            mean_grads = average_gradients(tower_grads)
            mean_pwc_grads = average_gradients(tower_pwc_grads)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?stabnet')
            update_pwc_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*?pwcnet')
            with tf.control_dependencies(update_ops):
                minimize_op = opt.apply_gradients(mean_grads)
            with tf.control_dependencies(update_pwc_ops):
                minimize_pwc_op = pwcnet_opt.apply_gradients(mean_pwc_grads)

        print('trainable variables:')
        print(vars_trainable)
        print('pwc trainable variables:')
        print(pwc_vars_trainable)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=200)
    lr_str = timestamp + ' ' + get_config(config) + ',gn:{}'.format(num_gpu)
    if not os.path.exists(checkpoint_path + lr_str):
        os.makedirs(checkpoint_path + lr_str)

    if args.pretrained:
        print('restore path from : ', checkpoint_path + args.lr_str + '/stab.ckpt-' + str(args.modeli))
        saver.restore(sess, checkpoint_path + args.lr_str + '/stab.ckpt-' + str(args.modeli))

    summary_ops = tf.summary.merge(summary)
    summary_writer = tf.summary.FileWriter(checkpoint_path + lr_str + '/summary', sess.graph)

    len_train = config.TRAIN.len_train
    n_epoch = 250

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    optimistic_restore(sess, pwc_opt.ckpt_path)

    for epoch in range(0, n_epoch):
        if epoch < pwc_freeze_epoch:
            pwc_lr_init = 0.0
            sess.run(tf.assign(pwc_lr_v, pwc_lr_init)) # freeze the optical flow net
            log = ' ** pwc net new learning rate: %f ' % (pwc_lr_init)
            print(log)

        if epoch >= pwc_freeze_epoch:
            pwc_lr_init = config.TRAIN.pwc_lr_init
            cur_lr = pwc_lr_init
            sess.run(tf.assign(pwc_lr_v, cur_lr))
            log = ' ** pwc net new learning rate: %f ' % (cur_lr)
            print(log)

        if epoch >= pwc_lr_stable_epoch:
            pwc_lr_init = config.TRAIN.pwc_lr_init
            cur_lr = linear_lr(pwc_lr_init, pwc_decay_ratio, epoch - pwc_lr_stable_epoch)
            sess.run(tf.assign(pwc_lr_v, cur_lr))
            log = ' ** pwc net new learning rate: %f ' % (cur_lr)
            print(log)

        if epoch >= lr_stable_epoch:
            lr_init = config.TRAIN.lr_init
            cur_lr = linear_lr(lr_init, decay_ratio, epoch - lr_stable_epoch)
            sess.run(tf.assign(lr_v, cur_lr))
            log = ' ** stab net new learning rate: %f' % (cur_lr)
            print(log)

        sys.stdout.flush()

        epoch_time = time.time()
        for it in range(int(len_train/batch_size)):
            errM, _, _, summary = sess.run([mse_loss, minimize_op, minimize_pwc_op, summary_ops])
            if (it+int(len_train/batch_size)*epoch)%10==0:
                summary_writer.add_summary(summary, it+int(len_train/batch_size)*epoch)

            print("Epoch [%2d/%2d] %4d time: %4.4fs, loss:  %5.5f" %
                  (epoch, n_epoch, it, time.time() - epoch_time, errM))
            sys.stdout.flush()

            epoch_time = time.time()

            if (it + int(len_train/batch_size)*epoch)%1000 == 0:
                saver.save(sess, checkpoint_path + lr_str + '/stab.ckpt', global_step=(it + int(len_train/batch_size)*epoch))

    coord.request_stop()
    coord.join(threads)
    sess.close()

def build_model_test(first_img, mid_img, end_img, training=True, trainable=True):
    first_img = tf.cast(first_img, tf.float32)* 1.0 / 255.0
    mid_img = tf.cast(mid_img, tf.float32) * 1.0 / 255.0
    end_img = tf.cast(end_img, tf.float32) * 1.0 / 255.0
    pred_img, debug_out = testing_stab_model(first_img, mid_img, end_img, training=training, trainable=trainable)
    pred_img = tf.clip_by_value(pred_img, 0, 1.)
    return pred_img, debug_out


def stabilize(args):
    # args.resize = True
    print("reading images")
    if args.img_dir[-4:] == '.mp4' or args.img_dir[-4:] == '.avi':
        from utils import vid2img_lists
        img_lists = vid2img_lists(args.img_dir)
    else:
        from utils import file2lists
        img_lists = file2lists(os.path.join(args.img_dir, 'img_lists.txt'))
        img_lists = [item_i for item_i in img_lists if item_i[-3:] == 'png']
        img_lists = sorted(img_lists)
        img_lists = [os.path.join(args.img_dir, item_i) for item_i in img_lists]
        img_lists = [cv2.imread(fn)[:,:,::-1] for fn in img_lists]
    raw_shape = img_lists[0].shape
    if args.resize:
        img_lists = [pad_img(img, pwc_opt.pyr_lvls) for img in img_lists]
    else:
        from utils import resize_img
        img_lists = [resize_img(img, pwc_opt.pyr_lvls) for img in img_lists]

    first_img_p, mid_img_p, end_img_p = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]), \
                                  tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]), \
                                  tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

    out_img_ts, debug_out_ts = build_model_test(first_img_p, mid_img_p, end_img_p, training=False, trainable=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path + args.lr_str + '/stab.ckpt-' + str(args.modeli))

    for iter in range(args.stab_iter):
        print("--------iter {}---------".format(iter))
        next_img_lists = []
        for k in range(args.skip): next_img_lists.append(img_lists[k])
        for k in range(args.skip, len(img_lists)-args.skip):
            cur_img = img_lists[k]
            first_img = img_lists[k-args.skip]
            end_img = img_lists[k+args.skip]
            cur_img_, first_img_, end_img_ = list(map(lambda x: np.expand_dims(x, 0), [cur_img, first_img, end_img]))
            out_img, debug_out = sess.run([out_img_ts, debug_out_ts], feed_dict={first_img_p: first_img_,
                                                      mid_img_p: cur_img_,
                                                      end_img_p: end_img_})
            out_img = out_img.squeeze()
            out_img = np.array(out_img*255.0).astype(np.uint8)
            next_img_lists.append(out_img)

            if args.debug:
                debug_img_lists_k = ['first_img', 'cur_img', 'end_img', 'out_img']
                debug_img_lists_v = [first_img[:,:,::-1], cur_img[:,:,::-1], end_img[:,:,::-1], out_img[:,:,::-1]]
                debug_img_lists = dict(zip(debug_img_lists_k, debug_img_lists_v))
                # write_imgs(debug_img_lists, k, args.debug_out_dir)

                [warped_first, warped_end, img_int, flow_pred0, flow_pred1, flow_pred2] = debug_out
                debug_flow_lists_k = ['first2end_flow', 'end2first_flow', 'mid2int_flow']
                debug_flow_lists_v = [flow_pred0[0], flow_pred2[0], flow_pred1[0]]
                debug_flow_lists = dict(zip(debug_flow_lists_k, debug_flow_lists_v))
                write_flows(debug_flow_lists, k, args.debug_out_dir)

        for k in range(len(img_lists)-args.skip, len(img_lists)):
            next_img_lists.append(img_lists[k])
        img_lists = next_img_lists

    if args.resize:
        img_lists = [unpad_img(img, raw_shape) for img in img_lists]
    else:
        from utils import back_resize_img
        img_lists = [back_resize_img(img, raw_shape) for img in img_lists]

    # import pdb;pdb.set_trace();
    if args.img_dir[-4:] == '.mp4':
        from utils import save2vid
        save2vid(img_lists, args.out_dir, args.img_dir)
    else:
        save_img_lists(img_lists, args.out_dir)

if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate, test, export, psnr')
    parser.add_argument('--pretrained',type=bool, default=False,help='False, True')
    parser.add_argument('--resize', type=bool, default=False, help='False, True, resize the img or pad the img')
    parser.add_argument('--loss_type', type=str, default='pixel_wise', help='feature_reconstruct, pixel_wise')
    parser.add_argument('--debug', type=bool, default=False, help='False, True, debug or not')
    parser.add_argument('--seed', type=int, default=66, help='a random seed')
    parser.add_argument('--lr_str', type=str, default='09-28-12:33 bn:True,opt:Adam,bs:8,ims:256,lr:0.001,gt:True,ks:3,gn:1',
                        help='checkpoint path')
    parser.add_argument('--modeli', type=int, default=-1, help='loaded model version')
    parser.add_argument('--out_dir', type=str, default='./test_out/examples_5_out', help='output path of the resulting video, either as a single file or as a folder')
    parser.add_argument('--img_dir', type=str, default='./test_in/examples_5', help='input image path')
    parser.add_argument('--skip', type=int, default=2, help='skip step')
    parser.add_argument('--stab_iter', type=int, default=5, help='stab iter')
    parser.add_argument('--debug_out_dir', type=str, default='debug_out_dir', help='debug_out_dir')
    args = parser.parse_args()

    if args.debug:
        args.stab_iter = 1

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'train':
        args.modeli = 4000
        train(args)
    if args.mode == 'evaluate':
        args.modeli = 129000
        stabilize(args)
