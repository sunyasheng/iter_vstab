from easydict import EasyDict as edict
import numpy as np
config = edict()
config.TRAIN = edict()
config.TEST = edict()

config.data_root = '/mnt/cephfs_new_wj/lab_ad_idea/sunyasheng/davis_dataset/DAVIS'

config.TRAIN.lr_init = 0.001
config.TRAIN.pwc_lr_init = 0.000001
config.TRAIN.beta1 = 0.9
config.TRAIN.beta2 = 0.999
# config.TRAIN.pretrained_vgg_path = '/mnt/cephfs/arnold/sunyasheng/arnold_projects/sepconv-tensorflow/pretrained_vgg/vgg_19.ckpt'
config.TRAIN.lr_decay = 0.1
config.TRAIN.checkpoint_path = './checkpoints/stab_checkpoints/'
# config.TRAIN.loss_type = 'pixel_wise'
config.TRAIN.loss_type = 'feature_reconstruct'
config.TRAIN.optim_type = 'Adam'
config.TRAIN.per_gpu_batch_size = 8
config.TRAIN.image_input_size = 256
config.TRAIN.batch_norm = True
config.TRAIN.kernel_size = 3
config.TRAIN.decay_ratio = 0.00001
config.TRAIN.pwc_decay_ratio = 0.00000001
config.TRAIN.len_train = 10431 - 75
config.TRAIN.gated = True
config.TRAIN.reg = 0.1
config.TRAIN.pwc_freeze_epoch = 25
config.TRAIN.pwc_lr_stable_epoch = config.TRAIN.pwc_freeze_epoch + 25
config.TRAIN.lr_stable_epoch = 25

config.TRAIN.tf_records_path = '/mnt/cephfs_new_wj/lab_ad_idea/sunyasheng/projects/iter_vstab/Davis_dataset'
config.TRAIN.vimeo_data_path = '/mnt/cephfs_new_wj/lab_ad_idea/sunyasheng/datasets/Vimeo90K/vimeo_triplet/tri_trainlist.txt'

config.TRAIN.data_path = '/mnt/cephfs_new_wj/lab_ad_idea/sunyasheng/davis_dataset/DAVIS/ImageSets/480p/train.txt'
config.TRAIN.vgg19_npy_path = './pretrained_vgg19/vgg19.npy'

config.TEST.data_path = '/mnt/cephfs_new_wj/lab_ad_idea/sunyasheng/davis_dataset/DAVIS/ImageSets/480p/val.txt'
config.TEST.res_dir = './checkpoints/stab_checkpoints/test_res/'
config.DEBUG = False

config.TRAIN.trans_pix = config.TRAIN.image_input_size // 16

# config.rand_H_max = np.array([[1.1, 0.1, config.TRAIN.trans_pix],
#                               [0.1, 1.1, config.TRAIN.trans_pix],
#                               [0.000002, 0.000002, 1]])
# config.rand_H_min = np.array([[0.9, -0.1, -config.TRAIN.trans_pix],
#                               [-0.1, 0.9, -config.TRAIN.trans_pix],
#                               [-0.000002, -0.000002, 1]])

config.rand_H_max = np.array([[1, 0, config.TRAIN.trans_pix],
                              [0, 1, config.TRAIN.trans_pix],
                              [0, 0, 1]])
config.rand_H_min = np.array([[1, 0, -config.TRAIN.trans_pix],
                              [0, 1, -config.TRAIN.trans_pix],
                              [0, 0, 1]])
#