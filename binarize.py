import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

I = cv2.imread(sys.argv[1])
plt.figure(figsize=(20,10))
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

def binarize(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    #return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #res_image = cv2.threshold(blur, 0, 1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    res_image = cv2.threshold(blur, 60, 1,cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    res_image = cv2.morphologyEx(res_image, cv2.MORPH_OPEN, kernel)
    #res_image = cv2.morphologyEx(res_image, cv2.MORPH_CLOSE, kernel)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res_image, 8, cv2.CV_32S)
    i_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    return (labels == i_component).astype(np.uint8)

I_b = binarize(I)
cv2.imwrite(sys.argv[2], I_b * 255)