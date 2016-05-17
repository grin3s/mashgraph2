import numpy as np
import cv2
import matplotlib.pyplot as plt
from skeletonizer import Skeletonizer, SkelGraph





def binarize(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (21, 21), 0)
    #return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    res_image = cv2.threshold(blur, 0, 1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #res_image = cv2.threshold(blur, 80, 1,cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    res_image = cv2.morphologyEx(res_image, cv2.MORPH_OPEN, kernel)
    res_image = cv2.morphologyEx(res_image, cv2.MORPH_CLOSE, kernel)
    return res_image

I = cv2.imread("../training/001_r.tif")
plt.figure(figsize=(20,10))
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
I_b = binarize(I)
# I_b = np.array([[1,1,1,1],
#                [1,1,1,1],
#                [1,1,1,1]])

# I_b = np.array([[0,0,0,0,0,0,0,0,0,1],
#                 [0,1,1,0,0,0,0,0,1,1],
#                 [1,1,1,1,1,1,1,1,1,0],
#                 [0,1,1,1,1,1,1,0,0,0],
#                 [0,0,0,1,1,1,1,0,0,0],
#                 [0,0,0,1,0,0,1,0,0,0],
#                 [0,0,0,1,0,0,1,0,0,0]])

#I_b = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])

skel_obj = Skeletonizer(I_b)

I_skel = skel_obj.find_skeleton()

print(I_skel)
graph = SkelGraph(skel_obj, I_skel)
pass
#plt.imshow(I_skel, cmap="gray")

