import cv2
import sys
# sys.path.insert(0, '../../')
from utils import read_img_lists, match_outlines, get_H
import numpy as np
from scipy.fftpack import fft, ifft
import os
# import cvbase as cvb


class StabEval:
    def __init__(self, opt):
        self.input_video = opt.input_video
        self.output_video = opt.output_video
        self.properties = dict()

        print('----------------reading images------------------')
        in_img_lists = read_img_lists(self.input_video)
        out_img_lists = read_img_lists(self.output_video)
        self.in_img_lists = in_img_lists
        self.out_img_lists = out_img_lists
        print('input frames: ', len(in_img_lists))
        print('output frames: ', len(out_img_lists))

        print('---------computing Hs from src to tgt-----------')
        self.Hs = self.cal_homes()

        print('---------computing Hs in tgt--------------------')
        self.tgt_Hs = self.cal_homes_tgt()

        # print('----------------cropping ratio------------------')
        # self.cropping_ratio()
        print('--------------distortion ratio------------------')
        self.distortion_ratio()

        print('--------------stability score-------------------')
        self.stability_score()

        pass

    def cal_homes_tgt(self):
        res_Hs = []
        for i, out_img in enumerate(self.out_img_lists):
            if i == 0: continue
            H = get_H(self.out_img_lists[i-1], self.out_img_lists[i])
            res_Hs.append(H)
        return res_Hs

    def cal_homes(self):
        res_Hs = []
        for i, (in_img, out_img) in enumerate(zip(self.in_img_lists, self.out_img_lists)):
            H = get_H(in_img, out_img)
            res_Hs.append(H)
        return res_Hs

    def cropping_ratio(self):

        for i, (in_img, out_img) in enumerate(zip(self.in_img_lists, self.out_img_lists)):
            # prev_gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
            # cur_gray = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
            # prev_pts = cv2.goodFeaturesToTrack(prev_gray,
            #                                   maxCorners=200,
            #                                   qualityLevel=0.01,
            #                                   minDistance=30,
            #                                   blockSize=3)
            # cur_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, None)
            #
            # # prev_pts = np.array([kp.pt for kp in prev_pts], dtype='float32').reshape(-1, 1, 2)
            #
            # assert prev_pts.shape == cur_pts.shape
            #
            # idx = np.where(status == 1)[0]
            # prev_pts = prev_pts[idx]
            # cur_pts = cur_pts[idx]
            #
            # H, mask = cv2.findHomography(prev_pts, cur_pts, cv2.RANSAC, 5.0)
            # warped_img = cv2.warpPerspective(in_img, H, dsize=(in_img.shape[1], in_img.shape[0]))
            warped_img = match_outlines(in_img, out_img)
            cv2.imwrite('warped_img_{}.png'.format(i), warped_img)
            # print(H)
        pass

    def distortion_ratio(self):
        for H in self.Hs:
            eig_v, _ = np.linalg.eig(H[:2, :2])
            # ratio = np.min(eig_v) / np.max(eig_v)
            # U, s, V = np.linalg.svd(H[:2, :2])
            s = np.abs(eig_v)
            ratio = np.min(s) / np.max(s)
            print(ratio)

    def stability_score(self):
        trans_vs = []
        for H in self.tgt_Hs:
            trans = H[:2, 2]
            trans_v = np.linalg.norm(trans, 2)
            trans_vs.append( trans_v)
        trans_vs = np.array(trans_vs)

        trans_vs_freq = fft(trans_vs)
        # trans_real = trans_vs_freq.real
        # trans_imag = trans_vs_freq.imag

        xf = np.arange(len(trans_vs))
        xf1 = xf
        xf2 = xf1[range(int(len(xf)/2))]

        yf = np.abs(trans_vs_freq)
        yf1 = yf / len(trans_vs_freq)
        yf2 = yf1[range(int(len(yf)/2))]

        yf = yf[1:]
        yf1 = yf1[1:]
        yf2 = yf2[1:]

        xf = xf[1:]
        xf1 = xf1[1:]
        xf2 = xf2[1:]

        import matplotlib.pyplot as plt

        plt.subplot(221)
        plt.plot(np.arange(len(trans_vs)), trans_vs, color='#7A378B')
        plt.title('time domain')

        plt.subplot(222)
        plt.plot(xf, yf, 'r')
        plt.title('freq domain')

        plt.subplot(223)
        plt.plot(xf1, yf1, 'g')
        plt.title('normalized freq domain')

        plt.subplot(224)
        plt.plot(xf2, yf2, 'b')
        plt.title('half normalized freq domain')

        plt.show()

    def print_properties(self):
        for k, v in self.properties.items():
            print(k, v)

if __name__ == '__main__':
    from options import Options
    opt = Options().parse()
    StabEval(opt)