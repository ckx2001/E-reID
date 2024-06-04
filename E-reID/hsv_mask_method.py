import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
import time
from numba import jit


#######################不考虑mask的情况下，对图像的HSV颜色空间分布进行统计，并返回对应的统计直方图##############
@jit
def cal_hsvHist_speed_nomask(image, mask):
    """
        使用np.where得到mask = 1的图片像素位置索引，，，并使用numba包加速函数运行
    """

    h_bins = 18
    s_bins = 25
    v_bins = 25

    h_hist = np.zeros(h_bins, np.float32)
    s_hist = np.zeros(s_bins, np.float32)
    v_hist = np.zeros(v_bins, np.float32)


    h_image = image[:, :, 0]
    # cv2.imshow("h_image", h_image)
    s_image = image[:, :, 1]
    # cv2.imshow("s_image", s_image)
    v_image = image[:, :, 2]

    # h_bin_edges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360]
    # s_bin_edges = [0,10,20,30,40,50,60,70,80,90,100]

    t0 = time.time()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            h_value = h_image[i][j]
            h_index = math.floor(h_value / 10)
            if h_index == 18:
                h_index = 17
            h_hist[h_index] = h_hist[h_index] + 1

            s_value = s_image[i][j]
            s_index = math.floor(s_value / 10)
            if s_index == 25:
                s_index = 24
            s_hist[s_index] = s_hist[s_index] + 1

            v_value = v_image[i][j]
            v_index = math.floor(v_value / 10)
            if v_index == 25:
                v_index = 24
            v_hist[v_index] = v_hist[v_index] + 1

    t1 = time.time()
    print("加速时间：", (t1 - t0))

    #直方图归一化---很重要，这是尺度不变性的根本
    h_hist_final = h_hist/(np.max(h_hist))
    s_hist_final = s_hist / (np.max(s_hist))
    v_hist_final = v_hist / (np.max(v_hist))
    # h_hist_final = (h_hist - np.min(h_hist)) / (np.max(h_hist) - np.min(h_hist))
    # s_hist_final = (s_hist - np.min(s_hist))/(np.max(s_hist) - np.min(s_hist))

    final_hist = np.hstack((h_hist_final, s_hist_final, v_hist_final)) #列相加 12列 + 10列
    #return final_hist
    return final_hist


#######################在mask的考虑下，对图像的HSV颜色空间分布进行统计，并返回对应的统计直方图##############
@jit
def cal_hsvHist_speed(image, mask):
    """
        使用np.where得到mask = 1的图片像素位置索引，，，并使用numba包加速函数运行
    """
    h_bins = 18
    s_bins = 25
    v_bins = 25

    h_hist = np.zeros(h_bins, np.float32)
    s_hist = np.zeros(s_bins, np.float32)
    v_hist = np.zeros(v_bins, np.float32)

    true_index = np.where(mask == 1)

    h_image = image[:, :, 0]
    # cv2.imshow("h_image", h_image)
    s_image = image[:, :, 1]
    # cv2.imshow("s_image", s_image)
    v_image = image[:, :, 2]

    # h_bin_edges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360]
    # s_bin_edges = [0,10,20,30,40,50,60,70,80,90,100]

    t0 = time.time()

    for i, j in zip(true_index[0], true_index[1]):
        h_value = h_image[i][j]
        h_index = math.floor(h_value / 10)
        if h_index == 18:
            h_index = 17
        h_hist[h_index] = h_hist[h_index] + 1

        s_value = s_image[i][j]
        s_index = math.floor(s_value / 10)
        if s_index == 25:
            s_index = 24
        s_hist[s_index] = s_hist[s_index] + 1

        v_value = v_image[i][j]
        v_index = math.floor(v_value / 10)
        if v_index == 25:
            v_index = 24
        v_hist[v_index] = v_hist[v_index] + 1

    t1 = time.time()
    print("加速时间：", (t1 - t0))

    #直方图归一化---很重要，这是尺度不变性的根本
    h_hist_final = h_hist/(np.max(h_hist))
    s_hist_final = s_hist / (np.max(s_hist))
    v_hist_final = v_hist / (np.max(v_hist))
    # h_hist_final = (h_hist - np.min(h_hist)) / (np.max(h_hist) - np.min(h_hist))
    # s_hist_final = (s_hist - np.min(s_hist))/(np.max(s_hist) - np.min(s_hist))

    final_hist = np.hstack((h_hist_final, s_hist_final, v_hist_final)) #列相加 12列 + 10列
    #final_hist = np.hstack((h_hist_final, s_hist_final))
    #return final_hist
    return final_hist


def hsv_mask(img1, img2, cur_np1, cur_np2) -> float:
    '''
        根据图像统计得到HSV的颜色直方图，并根据直方图用巴士系数法求直方图相似度
        效果受mask的效果影响
    '''
    #得到HSV颜色空间下的图片
    thresh = 0
    hsv_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    masks1 = np.where(cur_np1 > thresh, 1, 0)
    masks2 = np.where(cur_np2 > thresh, 1, 0)

    true_index = np.where(masks2 == 1)
    # fraction_motor = len(true_index)/(img2.shape[0]*img2.shape[1])

    #cv2.imshow("input", img1)
    #cv2.waitKey(0)

    if masks2.sum() == 0:
        similarity_speed = 0
    else:
        # 根据mask 统计图片的直方图

        #hist1_speed = cal_hsvHist_speed_nomask(hsv_image1, masks1)
        hist1_speed = cal_hsvHist_speed(hsv_image1, masks1)

        #hist2_speed = cal_hsvHist_speed_nomask(hsv_image2, masks2)
        hist2_speed = cal_hsvHist_speed(hsv_image2, masks2)

        #similarity = 1 - cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_BHATTACHARYYA)
        similarity_speed = 1 - cv2.compareHist(hist1_speed, hist2_speed, method=cv2.HISTCMP_BHATTACHARYYA)


    #img2[masks2] = [255, 255, 255]
    similarity_speed = round(similarity_speed, 3)
    #font = cv2.FONT_HERSHEY_DUPLEX
    #img2 = cv2.putText(img2, f"{similarity_speed}", (int(img2.shape[1]/2), int(img2.shape[0]/2)), font, 1, (255, 255, 255), 1)

    #cv2.imshow(str(similarity_speed), img2)
    #save_path = os.path.join("./save_images", img_name)
    #cv2.imwrite(save_path, img2)
    #print(similarity)
    #print(similarity_speed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return similarity_speed








































