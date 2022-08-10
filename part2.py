try:
    import cv2
    import scipy
    import pandas as pd
    from scipy.ndimage import zoom
    import os
    import json
    import glob
    import argparse
    import h5py
    from PIL import Image
    from skimage import measure
    from skimage import filters
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy as np
    from scipy import signal as sg, ndimage,misc
    from scipy.ndimage import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

FILTER_PATH = 'gtFine_trainvaltest/gtFine/train/new'
SRC_PATH = 'leftImg8bit/train/new'



def get_table() -> object:
    save_table = pd.read_hdf('part2NN/attention_results/attention_results.h5')
    pd.set_option('display.max_rows', None)
    return save_table


def crop_images_from_table():
    df_result = pd.DataFrame({'seq':[],
                        'x0': [],
                        'x1':[],
                        'y0': [],
                        'y1': [],
                        'zoom': [],
                        'col': [],
                        'is_ignore':[],
                              'is_true': []
                              })
    counter = 0
    for index, row in df.iterrows():
        if row['x'] is None:
            continue
        if str(row["path"]).startswith("bochum") or str(row["path"]).startswith("aachen") or str(row["path"]).startswith("bremen"):
            result_of_image = crop_image(row['path'], int(row['x']), int(row['y']), row['zoom'], index,row['col'])
            if result_of_image == 'True':
                is_true = 1
                is_ignore = 0
            elif result_of_image == 'Ignore':
                is_ignore = 1
                is_true = 0
            elif result_of_image == 'False':
                is_true = 0
                is_ignore = 0
            df_result=pd.concat([df_result, pd.DataFrame.from_records([{'path': row['path'].replace('leftImg8bit.png','') + row['col']+result_of_image[0]+str(index)+'.png' , 'x0': int(row['x'])+20, 'x1': int(row['x'])+20, 'y0': int(row['y'])-30,'y1': int(row['y'])+30, 'zoom': row['zoom'], 'col': row['col'],
                   'is_true':is_true,'is_ignore':is_ignore, 'seq':index}])])


            counter += 1

    df_result.to_hdf('data.h5', key='df_result')


def crop_image(img_path: str, x: int, y: int, zoom: int, index: int,color:str):
    x_size = 20
    y_size = 30
    im = Image.open(SRC_PATH+'/'+img_path)

    label_name = img_path.replace("leftImg8bit", "gtFine_color")

    im_label = Image.open(FILTER_PATH+'/'+label_name)

    pixel_arr = im.crop((x - x_size / zoom, y - y_size / zoom, x + x_size / zoom, y + y_size / zoom))

    im = im.resize((30, 40))
    num_of_orange_pixel_in_image = check_orange_in_relation_to_picture(im_label, x, y, zoom)
    save_arr=np.array(im_label)
    save_orange_pixel = np.where(save_arr == [250, 170, 30, 255])


    if len(save_orange_pixel[0])/num_of_orange_pixel_in_image*100 > 60:
        result = "True"
    elif len(save_orange_pixel[0])/num_of_orange_pixel_in_image*100 >= 60 and len(save_orange_pixel[0]) /\
            num_of_orange_pixel_in_image * 100 >= 40:
        result = "Ignore"
    else:
        result = "False"

    im.save("crop_images/" + img_path.replace('leftImg8bit.png','') + color+result[0]+str(index) + ".png", format="png")
    return result


def check_orange_in_relation_to_picture(image: Image, x, y, zoom):
    pixel_arr = np.array(image.convert('L'))
    labs = measure.label(pixel_arr)
    temp = labs[int(y)][int(x)]
    num_of_orange_pixel = np.count_nonzero(labs == temp)
    return num_of_orange_pixel


def change_size(img, x, y, zoom_factor):
    w, h = img.size
    zoom2 = zoom_factor*2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    w, h = img.size

    return img.resize((w, h), Image.Resampling.LANCZOS)


if __name__ == '__main__':
    df = get_table()
    crop_images_from_table()



