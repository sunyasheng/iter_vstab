from easydict import EasyDict as edict

pwc_opt = edict()
pwc_opt.pyr_lvls = 6
pwc_opt.flow_pred_lvl = 2
pwc_opt.dbg = False
pwc_opt.search_range = 4
pwc_opt.use_dense_cx = True
pwc_opt.use_res_cx = True
# pwc_opt.x_shape = [2, 384, 448, 3]
# pwc_opt.y_shape = [384, 448, 2]
pwc_opt.ckpt_path = '../tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
