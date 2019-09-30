import cv2
import sys
# sys.path.insert(0, '../../')
from utils import read_img_lists, match_outlines, get_H
import numpy as np
from scipy.fftpack import fft, ifft
import os
import math
# import cvbase as cvb
import matplotlib.pyplot as plt


class StabEval:
    def __init__(self, opt):
        self.input_dir = opt.input_video
        self.output_dir = opt.output_video
        self.properties = dict()

        print('----------------reading images------------------')
        in_img_lists = read_img_lists(self.input_dir)
        out_img_lists = read_img_lists(self.output_dir)
        self.in_img_lists = in_img_lists
        self.out_img_lists = out_img_lists
        print('input frames: ', len(in_img_lists))
        print('output frames: ', len(out_img_lists))

        # print('---------computing Hs from src to tgt-----------')
        # self.Hs = self.cal_homes()

        # print('---------computing Hs in tgt--------------------')
        # self.tgt_Hs = self.cal_homes_tgt()

        # print('----------------cropping ratio------------------')
        # self.cropping_ratio()
        # print('--------------distortion ratio------------------')
        # self.distortion_ratio()

        print('--------------stability score---------------------')
        SS_t, SS_r = self.stability_score()
        log = str.format('{0:.4f}', (SS_t + SS_r) / 2) + ' | ' + str.format('{0:.4f}', SS_t) + ' | ' + str.format(
            '{0:.4f}', SS_r)
        print(log)
        with open(opt.stable_logfile, 'a+') as f:
            f.write(log)
            f.write('\n')

        # print('----distortion ratio and cropping ratio-----------')
        # CR_seq, DV_seq = self.DCR()
        #
        # print('***Cropping ratio (Avg, Min):')
        # print(str.format('{0:.4f}', np.min([np.mean(CR_seq), 1])) + ' | '
        #     + str.format('{0:.4f}', np.min([CR_seq.min(), 1])))
        # print('***Distortion value:')
        # print(str.format('{0:.4f}', np.absolute(DV_seq.min())))

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
            warped_img = match_outlines(in_img, out_img)
            cv2.imwrite('warped_img_{}.png'.format(i), warped_img)

    def DCR(self):
        # for H in self.Hs:
        #     eig_v, _ = np.linalg.eig(H[:2, :2])
        #     # ratio = np.min(eig_v) / np.max(eig_v)
        #     # U, s, V = np.linalg.svd(H[:2, :2])
        #     s = np.abs(eig_v)
        #     ratio = np.min(s) / np.max(s)
        #     print(ratio)
        # Apply the homography transformation if we have enough good matches
        MIN_MATCH_COUNT = 10  # 10

        ratio = 0.7  # 0.7
        thresh = 5.0  # 5.0
        bf = cv2.BFMatcher()

        CR_seq = np.asarray([1])
        DV_seq = np.asarray([1])

        for i, (in_img, out_img) in enumerate(zip(self.in_img_lists, self.out_img_lists)):

        # for i, out_img in enumerate(self.out_img_lists):
        #     if i == 0: continue
            # Detect the SIFT key points and compute the descriptors for the two images
            sift = cv2.xfeatures2d.SURF_create()
            keyPoints1, descriptors1 = sift.detectAndCompute(in_img, None)
            keyPoints1o, descriptors1o = sift.detectAndCompute(out_img, None)

            # Match the descriptors
            matches = bf.knnMatch(descriptors1, descriptors1o, k=2)

            # Select the good matches using the ratio test
            goodMatches = []

            for m, n in matches:
                if m.distance < ratio * n.distance:
                    goodMatches.append(m)

            if len(goodMatches) > MIN_MATCH_COUNT:
                # Get the good key points positions
                sourcePoints = np.float32([keyPoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
                destinationPoints = np.float32([keyPoints1o[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

                # Obtain the homography matrix
                M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC,
                                             ransacReprojThreshold=thresh)
            # end

            # Obtain Scale, Translation, Rotation, Distortion value
            sx = M[0, 0]
            sy = M[1, 1]
            scaleRecovered = math.sqrt(sx * sy)

            w, _ = np.linalg.eig(M[0:2, 0:2])
            w = np.sort(w)[::-1]
            DV = w[1] / w[0]
            # pdb.set_trace()

            CR_seq = np.concatenate((1.0 / CR_seq, [scaleRecovered]), axis=0)
            DV_seq = np.concatenate((DV_seq, [DV]), axis=0)

        CR_seq = np.delete(CR_seq, 0)
        DV_seq = np.delete(DV_seq, 0)

        return CR_seq, DV_seq

    def stability_score(self):
        sift = cv2.xfeatures2d.SURF_create()
        bf = cv2.BFMatcher()
        MIN_MATCH_COUNT = 10  # 10

        ratio = 0.7  # 0.7
        thresh = 5.0  # 5.0

        P_seq = []
        Pt = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        keyPoints1o, descriptors1o = sift.detectAndCompute(self.out_img_lists[0], None)
        for i, out_img in enumerate(self.out_img_lists):
            if i == 0: continue
            keyPoints2o, descriptors2o = sift.detectAndCompute(self.out_img_lists[i], None)
            matches = bf.knnMatch(descriptors1o, descriptors2o, k=2)

            goodMatches = []

            for m, n in matches:
                if m.distance < ratio * n.distance:
                    goodMatches.append(m)

            if len(goodMatches) > MIN_MATCH_COUNT:
                # Get the good key points positions
                sourcePoints = np.float32([keyPoints1o[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
                destinationPoints = np.float32([keyPoints2o[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

                # Obtain the homography matrix
                M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC,
                                             ransacReprojThreshold=thresh)
            P_seq.append(np.matmul(Pt, M))
            Pt = np.matmul(Pt, M)

            keyPoints1o, descriptors1o = keyPoints2o, descriptors2o

        P_seq_t = np.asarray([1])
        P_seq_r = np.asarray([1])

        # pdb.set_trace()
        for Mp in P_seq:
            sx = Mp[0, 0]
            sy = Mp[1, 1]
            c = Mp[0, 2]
            f = Mp[1, 2]
            # w, _ = np.linalg.eig(Mp[0:2,0:2])
            # w = np.sort(w)[::-1]
            # DV = w[1]/w[0]
            transRecovered = math.sqrt(c * c + f * f)
            thetaRecovered = math.atan2(sx, sy) * 180 / math.pi
            # thetaRecovered = DV
            P_seq_t = np.concatenate((P_seq_t, [transRecovered]), axis=0)
            P_seq_r = np.concatenate((P_seq_r, [thetaRecovered]), axis=0)

        P_seq_t = np.delete(P_seq_t, 0)
        P_seq_r = np.delete(P_seq_r, 0)

        # FFT
        fft_t = np.fft.fft(P_seq_t)
        fft_r = np.fft.fft(P_seq_r)
        fft_t = abs(fft_t) ** 2
        fft_r = abs(fft_r) ** 2
        # freq = np.fft.fftfreq(len(P_seq_t))
        # plt.plot(freq, abs(fft_r)**2)
        # plt.show()
        # print(abs(fft_r)**2)
        # print(freq)
        fft_t = np.delete(fft_t, 0)
        fft_r = np.delete(fft_r, 0)
        fft_t = fft_t[:int(len(fft_t) / 2)]
        fft_r = fft_r[:int(len(fft_r) / 2)]

        SS_t = np.sum(fft_t[:5]) / np.sum(fft_t)
        SS_r = np.sum(fft_r[:5]) / np.sum(fft_r)

        return SS_t, SS_r


    def print_properties(self):
        for k, v in self.properties.items():
            print(k, v)

if __name__ == '__main__':
    from options import Options
    opt = Options().parse()
    StabEval(opt)