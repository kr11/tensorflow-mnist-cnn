import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy import ndimage


def load_data(data_dir, n_label, validation_rate=0.2, is_expanding=True):
    '''Fetch all data into a list'''
    '''TODO: 1. You may make it more memory efficient if there is a OOM problem on
    you machine. 2. You may use data augmentation tricks.'''
    train_xs = []
    train_ys = []
    validation_xs = []
    validation_ys = []
    label_dirs = os.listdir(data_dir)
    label_dirs.sort()
    for _label_dir in label_dirs:
        if not _label_dir.startswith('label'):
            continue
        print('loaded {}'.format(_label_dir))
        category = int(_label_dir[5:])
        if category >= n_label:
            continue
        label = np.zeros(n_label)
        label[category] = 1
        img_names = os.listdir(os.path.join(data_dir, _label_dir))
        n_img = len(img_names)
        np.random.shuffle(img_names)
        validation_size = int(n_img * validation_rate)
        for img_name in img_names[0:validation_size]:
            im_ar = read_image(os.path.join(data_dir, _label_dir, img_name))
            validation_xs.append(im_ar)
            validation_ys.append(label)
        # expanding
        s = 0
        for img_name in img_names[validation_size:]:
            s += 1
            if s % 5 == 0:
                print(s)
            im_ar = read_image(os.path.join(data_dir, _label_dir, img_name))
            train_xs.append(im_ar)
            train_ys.append(label)
            if is_expanding:
                expanding(im_ar, label, train_xs, train_ys)
    return train_xs, train_ys, validation_xs, validation_ys


def read_image(image_name):
    im_ar = cv2.imread(image_name)
    im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
    im_ar = np.asarray(im_ar)
    im_ar = preprocess(im_ar)
    return im_ar


def expanding(im_ar, label, train_xs, train_ys, expanding_size=2, flip_prob=0.5):
    """
    :return:
    """
    bg_value = np.median(im_ar)
    for i in range(expanding_size):
        angle = np.random.randint(-15, 15, 1)
        new_img = ndimage.rotate(im_ar, angle, reshape=False, cval=bg_value)

        # shift the image with random distance
        shift = np.random.randint(-2, 2, 2)
        new_img = ndimage.shift(new_img, np.append(shift, 0), cval=bg_value)

        # flip with certain probability
        if np.random.rand() > flip_prob:
            new_img = np.fliplr(new_img)
        if np.random.rand() > flip_prob:
            new_img = np.flipud(new_img)
        train_xs.append(new_img)
        train_ys.append(label)


def preprocess(im_ar):
    """
    Resize raw image to a fixed size, and scale the pixel intensities.
    :param im_ar:
    :return:
    """
    im_ar = cv2.resize(im_ar, (224, 224))
    im_ar = im_ar / 255.0

    return im_ar


if __name__ == '__main__':
    load_data('dset1', 10)
    print("finished")
