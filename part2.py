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
    print(save_table)
    return save_table


def crop_images_from_table():
    counter=0
    for index, row in df.iterrows():
        if str(row["path"]):
            crop_image(row['path'], int(row['x']), int(row['y']), row['zoom'],counter)
            counter+=1


def crop_image(img_path: str, x: int, y: int, zoom: int, index: int):
    im = Image.open(SRC_PATH+'/'+img_path)
    label_name = img_path.replace("leftImg8bit", "gtFine_color")

    im_label = Image.open(FILTER_PATH+'/'+label_name)
    width, height = im.size
    im_of_label = im_label.crop((x-15, y-30, x+15, y+30))
    pixel_arr = np.array(im_of_label)
    save_orange_pixel = np.where(pixel_arr == [250, 170, 30, 255])
    print(len(save_orange_pixel[0]))
    if len(save_orange_pixel[0]) >4000:
        im.crop((x - 15, y - 30, x + 15, y + 30)).save("crop_images/crop" + str(index) + ".png", format="png")
        im_of_label.save("crop_images/crop" + str(index) + "label.png", format="png")


if __name__ == '__main__':
    df = get_table()
    crop_images_from_table()

