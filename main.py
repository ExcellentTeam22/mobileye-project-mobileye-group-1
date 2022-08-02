import cv2
import scipy

try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg, ndimage,misc
    from scipy.ndimage import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def convert_image_to_array(path: str) :
    implt = plt.imread(path)
    maplt = implt[:, :, 0]
    plt.imshow(maplt)
    return implt


    # image = cv2.imread(path)
    # # lower = np.array([0, 50, 50])
    # # upper = np.array([10, 255, 255])
    # # mask = cv2.inRange(image, lower, upper)
    # mask = image[:, :, [2]]
    # plt.imshow(mask)
    # return mask


def one_image():
    #kernel = convert_image_to_array("kernel.png")

    kernel = (plt.imread("kernel.png") / 255)
    kernel = kernel[:, :, 0]
    kernel -= np.mean(kernel)
    print(kernel)
    image = plt.imread("test_images\\berlin_000522_000019_leftImg8bit.png")
    image = image[:, :, 0]
    print(image)
    filter_image = scipy.ndimage.convolve(image, kernel)
    plt.imshow(filter_image)
    plt.show(block=True)
    print(filter_image)
    print("max")
    fig = plt.figure()
    ax1 = fig.add_subplot()  # left side
    ax2 = fig.add_subplot()  # right side
    result = ndimage.maximum_filter(filter_image, size=20)
    save_anc=np.where(result > 0.1)
    ax2.imshow(result)
    print(result)
    plt.show()
    return(save_anc)




def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel = (plt.imread("kernel.png") / 255)
    kernel = kernel[:, :, 0]
    kernel -= np.mean(kernel)
    print(kernel)
    #image = plt.imread(c_image)
    image=c_image
    image = image[:, :, 0]
    print(image)
    filter_image = scipy.ndimage.convolve(image, kernel)
    #plt.imshow(filter_image)
    #plt.show(block=True)
    print(filter_image)
    print("max")
    result = ndimage.maximum_filter(filter_image, size=10)
    save_anc = np.where(result > 0.1)
    #ax2.imshow(result)
    print(result)
    plt.show()
    return save_anc[1], save_anc[0], [], []


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(plt.imread(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    #show_image_and_gt(image, objects, fig_num)

    plt.figure(56)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(image)
    plt.figure(57)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(image)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)



def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "test_images"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    #one_image()
    main()
    #test_find_tfl_lights("test_images\\berlin_000522_000019_leftImg8bit.png")


