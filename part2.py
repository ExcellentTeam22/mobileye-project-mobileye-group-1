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

FILTER_PATH = 'gtFine_trainvaltest/gtFine/train/aachen'
SRC_PATH = 'leftImg8bit/train/aachen'



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
        if str(row["path"]):
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
 #   im=change_size(im1,x,y,zoom)


    label_name = img_path.replace("leftImg8bit", "gtFine_color")

    im_label1 = Image.open(FILTER_PATH+'/'+label_name)

    width, height = im.size


    pixel_arr = np.array(im_label1)
    im_label = clipped_zoom(pixel_arr, zoom)

    data = Image.fromarray(im_label)
   # plt.imshow(data)
    #plt.show()
    im_label=data

    im_of_label = im_label.crop((x-x_size, y-y_size, x+x_size, y+y_size))


    #pixel_arr = clipped_zoom(pixel_arr_without_zoom ,zoom)

    save_orange_pixel = np.where(pixel_arr == [250, 170, 30, 255])

    num_of_orange_pixel_in_image = check_orange_in_relation_to_picture(im_label, x, y, zoom)

    if len(save_orange_pixel[0])/num_of_orange_pixel_in_image*100 > 60:
        result = "True"
    elif len(save_orange_pixel[0])/num_of_orange_pixel_in_image*100 >= 60 and len(save_orange_pixel[0]) /\
            num_of_orange_pixel_in_image * 100 >= 40:
        result = "Ignore"
    else:
        result = "False"

    im.crop((x - x_size, y - y_size, x + x_size, y + y_size)).save("crop_images/" + img_path.replace('leftImg8bit.png','') + color+result[0]+str(index) + ".png", format="png")
    im_of_label.save("crop_images/" + img_path.replace('leftImg8bit.png','') + color+result[0]+str(index) + "label.png", format="png")
    return result


def check_orange_in_relation_to_picture(image: Image, x, y, zoom):
    pixel_arr = np.array(image.convert('L'))

    #pixel_arr = clipped_zoom( pixel_arr_without_zoom , zoom)

    labs = measure.label(pixel_arr)
    temp = labs[int(y)][int(x)]
    num_of_orange_pixel = np.count_nonzero(labs == temp)
    return num_of_orange_pixel

    #def check_valid(x,y,add_to_x,add_to_y,image_height,image_width):
    #  if x + add_to_x >image_width:

def change_size(img,x,y, zoom_factor):
    w, h = img.size
    zoom2 = zoom_factor*2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    w, h = img.size

    return img.resize((w, h), Image.Resampling.LANCZOS)





def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out




if __name__ == '__main__':
    df = get_table()
    crop_images_from_table()



