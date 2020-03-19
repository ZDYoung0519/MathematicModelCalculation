import numpy as np
import cv2
from matplotlib import pyplot as plt


def GammaTransform(src, c=1, gamma=1):
    if c == 1 and gamma == 1:
        dst = src.copy()
    else:
        src = src.copy().astype(np.float32)
        dst = c * (src ** gamma)
        cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
        dst = dst.astype(np.uint8)
    return dst


def LocalGrayTransform1(src, A, B, min_val=0, max_val=255):
    src = src.copy()
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if A <= i <= B:
            lut[i] = max_val
        else:
            lut[i] = min_val
    dst = cv2.LUT(src, lut)
    dst = dst.astype(np.uint8)
    return dst


def LocalGrayTransform2(src, A, B, max_val=255):
    src = src.copy()
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if A <= i <= B:
            lut[i] = max_val
        else:
            lut[i] = i
    dst = cv2.LUT(src, lut)
    dst = dst.astype(np.uint8)
    return dst


def GetHist(img):
    hist = np.zeros(256, dtype=np.float32)
    for i in range(256):
        hist[i] = np.sum(img == i)
    return hist


def HistogramEqualizationGray(src):
    if len(src.shape) == 3:
        raise ValueError('Please input gray image (not color).')
    lut = np.zeros(256, dtype=np.float32)
    n = np.prod(src.shape)
    sum_p = 0
    for i in range(256):
        p = np.sum(src == i) / n
        sum_p += p
        lut[i] = 255 * sum_p
    dst = cv2.LUT(src, lut)
    dst = dst.astype(np.uint8)
    return dst


def HistogramEqualizationColor(src):
    if len(src.shape) != 3:
        raise ValueError('Please input RGB image (not gray).')
    dst = np.zeros(src.shape, dtype=np.float32)
    for i in range(3):
        src_i = src[:, :, i]
        dst_i, _, _ = HistogramEqualizationGray(src_i)
        dst[:, :, i] = dst_i
    return dst

