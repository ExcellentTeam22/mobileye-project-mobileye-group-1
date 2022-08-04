
try:
    import cv2
    import scipy
    import pandas as pd
    import os
    import json
    import glob
    import argparse
    import h5py


    import numpy as np


    from scipy import signal as sg, ndimage,misc
    from scipy.ndimage import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


if __name__ == '__main__':
    df = pd.read_hdf('attention_results.h5')
    pd.set_option('display.max_rows', None)
    print(df)
