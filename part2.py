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


FILTER_PATH = "gtFine_trainvaltest/gtFine/train/aachen"
SRC_PATH = "leftImg8bit/train/aachen"


def get_table() -> object:
    save_table = pd.read_hdf('attention_results.h5')
    pd.set_option('display.max_rows', None)
    return object


def crop_images_from_table():
    for index, row in df.iterrows():
        if str(row["path"]).startswith("bochum"):
            crop_image(row['path'], int(row['x']), int(row['y']), row['zoom'])


def crop_image(img_path: str, x: int, y: int, zoom: int):
    im = Image.open(SRC_PATH+'/'+img_path)
    width, height = im.size
    # Setting the points for cropped image
    # Cropped image of above dimension
    # (It will not change original image)
    #im1 = im.crop((left, top, right, bottom))

    im1 = im.crop((x-15, y-30, x+15, y+30))

    # Shows the image in image viewer

    plt.imshow(im1)  # print image with opacity filter
    plt.show()


if __name__ == '__main__':
    df = get_table()
    crop_images_from_table()

