import os

import scipy
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np

# #reading the image
# img = imread('dataset/n01855672/n0185567200000003.jpg')
# # imshow(img)
# # plt.show()
# # print(img.shape)
#
# #resizing image
# resized_img = resize(img, (64,64))
# # imshow(resized_img)
# # print(resized_img.shape)
#
# #creating hog features
# fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
#                     cells_per_block=(2, 2), visualize=True, multichannel=True)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
#
# ax1.imshow(resized_img, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
#
# plt.show()


def extract_features(image_path):
    image = imread(image_path)
    resized_image = resize(image, (64,64))
    #creating hog features
    fd, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd, hog_image

def batch_extractor():
    directories = [os.path.join("dataset/", p) for p in sorted(os.listdir("dataset/"))]
    files = []
    flattened_files = []

    for dir in directories:
        files.append([os.path.join(dir + "/", p) for p in sorted(os.listdir(dir + "/"))])

    for x in files:
        for y in x:
            flattened_files.append(y)
    result = {}
    for f in flattened_files:
        name = f.split('/')[-2:]
        name = '/'.join([str(elem) for elem in name])
        result[name], _ = extract_features(f)
    return result


def compute_distance(query_image_features, feature_vectors):

    query_image_features = query_image_features.reshape(1, -1)

    return scipy.spatial.distance.cdist(feature_vectors, query_image_features, 'cosine').reshape(-1)


def match(query_image, features):
    query_image_features, _ = extract_features(query_image)
    query_image_features = np.array(query_image_features)

    names = np.array(list(features.keys()))
    feature_vectors = np.array(list(features.values()))


    distances = compute_distance(query_image_features, feature_vectors)

    #top 5
    nearest_ids = np.argsort(distances)[1:6].tolist()
    nearest_img_paths = names[nearest_ids].tolist()

    return nearest_img_paths, distances[nearest_ids].tolist()

def show_img(path):
    img = imread(path)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    res = batch_extractor()
    chosen_query_image = 'dataset/n01855672/n0185567200000003.jpg'
    img_paths,_ = match(chosen_query_image, res)
    print(img_paths)
    show_img(chosen_query_image)
    for img in img_paths:
        show_img('dataset/' + img)



