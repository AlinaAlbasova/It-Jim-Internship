import os
import numpy as np
from skimage.io import imread
from keras.utils.np_utils import to_categorical



classes = os.listdir("dataset/")
class_numbers = range(0,16)


def prep_data(images):
    """
     This function preprocesses input images. Firstly, it creates two arrays for images and labels, responsively.
    Then, it read images in the mentioned path, transform them to grayscale and put it to the array.
    Next, based on the image name, we create an appropriate label and store it in another array.
    After all, we return prepared dataset.
    :param images: path to the folder with images that will be processed
    :return: array with grayscale images and array with classes of this images
    """
    m = len(images)
    ROWS, COLS, CHANNELS = 84,84,3

    X = np.ndarray((m,ROWS,COLS,CHANNELS))
    y = np.zeros((m,1))

    for i, img_file in enumerate(images):
        image = imread(img_file, as_gray=False)
        X[i,:] = image
        for idx, cls in enumerate(classes):
            if cls in img_file.lower():
                y[i,0] = class_numbers[idx]

    y = y.reshape(-1)
    # X = np.expand_dims(X, axis=3)
    # y_one_hot = to_categorical(y)

    return X, y
