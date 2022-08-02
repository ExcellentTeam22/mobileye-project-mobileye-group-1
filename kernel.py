

import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# #
#
#
# def convert_image_to_array(path: str) :
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower = np.array([0, 50, 50])
#     upper = np.array([10, 255, 255])
#     mask = cv2.inRange(image, lower, upper)
#     cv2.imshow('mask', mask)
#     return np.array(mask)
#
#
# image = cv2.imread("test_images\\berlin_000521_000019_leftImg8bit.png")
# result = image.copy()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower = np.array([0,50,50])
# upper = np.array([10,255,255])
# mask = cv2.inRange(image, lower, upper)
#
# print(mask)
#
# cv2.imshow('mask', mask)
# kernel = np.array(mask)
#
# img = cv2.filter2D(convert_image_to_array("test_images\\berlin_000521_000019_leftImg8bit.png"), -1, convert_image_to_array("kernel.png"))
# plt.imshow(img)
# plt.show(block=True)
#
# #print(kernel)
# cv2.waitKey()

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

#
# x = np.array([[1,2,3],[2,3,4]])
# x.mean()



import numpy as np
from scipy.ndimage import maximum_filter
import scipy
import matplotlib.pyplot as plt
def main():
    kernel = (plt.imread("kernel.png")/255)
    kernel = kernel[:, :, 0]
    kernel -= np.mean(kernel)
    print(kernel)
    img = plt.imread("test_images\\berlin_000522_000019_leftImg8bit.png")
    img = img[:, :, 0]
    array = scipy.ndimage.convolve(img, kernel)
    plt.imshow(array)
    print("done")
if __name__ == '__main__':
    main()