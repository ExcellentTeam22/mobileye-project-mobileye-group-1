
try:
    import cv2
    import scipy
    import pandas as pd
    import os
    import json
    import glob
    import argparse
    import h5py
    from PIL import Image
    import numpy as np


    from scipy import signal as sg, ndimage,misc
    from scipy.ndimage import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise




def crop_image(img_path: str, x: int, y: int, zoom: int):
    im = Image.open("r"+img_path)
    width, height = im.size
    # Setting the points for cropped image
    # Cropped image of above dimension
    # (It will not change original image)
    #im1 = im.crop((left, top, right, bottom))

    im1 = im.crop(x-15, y+30, x+15, y-30)

    # Shows the image in image viewer
    im1.show()


if __name__ == '__main__':
    df = pd.read_hdf('attention_results.h5')
    pd.set_option('display.max_rows', None)
    print(df)

    crop_image()
