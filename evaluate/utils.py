# import cvbase as cvb
import glob
import os
import cv2
import numpy as np
import math

def read_img_lists(video_path):
    # print(video_path)
    img_paths = glob.glob(os.path.join(video_path, '*.png'))
    # print(img_paths)
    res_imgs = []
    for i in range(len(img_paths)):
        res_imgs.append(cv2.imread(img_paths[i]))
    return res_imgs

def get_H(orig_image, skewed_image):
    orig_image = np.array(orig_image)
    skewed_image = np.array(skewed_image)
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
    except Exception:
        surf = cv2.SIFT(400)
    kp1, des1 = surf.detectAndCompute(orig_image, None)
    kp2, des2 = surf.detectAndCompute(skewed_image, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    else:
        print('not enough match count! use identity matrix')
        return np.eye(3)

def match_outlines(orig_image, skewed_image):
    orig_image = np.array(orig_image)
    skewed_image = np.array(skewed_image)
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
    except Exception:
        surf = cv2.SIFT(400)
    kp1, des1 = surf.detectAndCompute(orig_image, None)
    kp2, des2 = surf.detectAndCompute(skewed_image, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        # ss = M[0, 1]
        # sc = M[0, 0]
        # scaleRecovered = math.sqrt(ss * ss + sc * sc)
        # thetaRecovered = math.atan2(ss, sc) * 180 / math.pi

        # deskew image
        im_out = cv2.warpPerspective(orig_image, np.linalg.inv(M),
                                     (orig_image.shape[1], orig_image.shape[0]))
        return im_out

    else:
        print('not enough match count!')
        return orig_image