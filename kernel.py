

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
#

image = cv2.imread("kernel.png")
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0,50,50])
upper = np.array([10,255,255])
mask = cv2.inRange(image, lower, upper)

result = cv2.bitwise_and(result, result, mask=mask)
result= cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
print(result)

cv2.imshow('mask', mask)
cv2.imshow('result', result)
kernel = np.array(result)



#print(kernel)
cv2.waitKey()

#
# image = Image.open("kernel.png").convert("L")
# plt.imshow(image)
# plt.show(block=True)
# kernel = np.array(image)
# print(kernel)







# im = cv2.imread("kernel.png")
#
# # To Grayscale
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#
#
# # To Black & White
# im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]
# plt.imshow(im)
# plt.show(block=True)
# cv2.waitKey()
# print(im)