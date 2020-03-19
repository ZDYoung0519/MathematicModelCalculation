import cv2
from utils.transformations import *
import matplotlib.pyplot as plt


img_path = 'footage/u=3791962068,4292654682&fm=15&gp=0.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 幂次变换
c, gamma = 1, [1, 2, 3, 4]
for i, gm in enumerate(gamma):
    new_img = GammaTransform(img, 1, gm)
    cv2.imshow('Gamma transformation, c={:d}, gamma={:d}.'.format(c, gm), new_img)

# 局部灰度变换
new_img = LocalGrayTransform1(img, 190, 220, 190, 220)
cv2.imshow('LocalGrayTransform1', new_img)
new_img = LocalGrayTransform2(img, 190, 220, 220)
cv2.imshow('LocalGrayTransform2', new_img)

# 直方图均衡化
new_img = HistogramEqualizationGray(img)
ori_hist, after_hist = GetHist(img), GetHist(new_img)
# new_img = HistogramEqualizationColor(img)
cv2.imshow('Historgram equlization', new_img)
plt.figure('Origin histogram')
plt.bar(np.arange(256), ori_hist)
plt.figure('New histogram')
plt.bar(np.arange(256), after_hist)

# 滤波
# 中值滤波
new_img = cv2.medianBlur(img, 3)
cv2.imshow('Median blur', new_img)
# 均值滤波
new_img = cv2.blur(img, (3, 3))
cv2.imshow('Mean blur', new_img)
#


plt.show()
cv2.waitKeyEx()

