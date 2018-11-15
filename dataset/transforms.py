import random
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import math
import random
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.model_selection import train_test_split
from torchvision import transforms
from matplotlib import pyplot as plt

def random_horizontal_flipN(imgs, u=0.5):
    """
    random horiziontal flip
    :param imgs:
    :param u:
    :return:
    """
    if random.random() < u:
        for n, img in enumerate(imgs):
            imgs[n] = np.fliplr(img)
    return imgs


def resizeN(imgs, new_size):
    """
    resize input images
    :param imgs:
    :param new_size:
    :return:
    """
    for i, img  in enumerate(imgs):
        imgs[i] = resize(img, new_size, mode='reflect', preserve_range=True, anti_aliasing=False)
    return imgs


def elastic_transform(images, alpha, sigma, random_state=None, u = 0.5):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random.random() > u:
        return images
    # print('Distorted')

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = images[0].shape         #;print(shape)

    """Random affine"""
    # center_square = np.float32(shape_size) // 2
    # square_size = min(shape_size) // 3
    # pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
    #                    center_square - square_size])
    # pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # M = cv2.getAffineTransform(pts1, pts2)
    # for n, image in enumerate(images):
    #     images[n] = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    # print(images[0].shape)


    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
    #print(indices[0].shape)
    distorted_images = []
    for n, image in enumerate(images):
        distorted_images.append(map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape))
        # print(distorted_images[n].shape)
        # cv2.imwrite('{}.png'.format(n), map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape))
    return distorted_images

def random_hue(image, hue_limit=(-0.1, 0.1), u=0.5):
    if random.random() < u:
        h = int(random.uniform(hue_limit[0], hue_limit[1]) * 180)
        # print(h)

        image = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    return image

def random_gray(image, u=0.5):
    if random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(image * coef, axis=2)
        image = np.dstack((gray, gray, gray))
    return image


def random_brightness(image, limit=(-0.3, 0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        image = alpha * image
        image = np.clip(image, 0., 1.)
    return image


def random_contrast(image, limit=(-0.3, 0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha * image + gray
        image = np.clip(image, 0., 1.)
    return image


def random_saturation(image, limit=(-0.3, 0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = image * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        image = alpha * image + (1.0 - alpha) * gray
        image = np.clip(image, 0., 1.)
    return image

# transform image -----------------------------
def random_horizontal_flip(image, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 1)
    return image


def fix_crop(image, roi=(0, 0, 256, 256)):
    x0, y0, x1, y1 = roi
    image = image[y0:y1, x0:x1, :]
    return image


def fix_resize(image, w, h):
    image = cv2.resize(image, (w, h))
    return image


# transform image and label -----------------------------

def random_horizontal_flipN(images, threshold=0.5):
    if random.random() < threshold:
        for n, image in enumerate(images):
            images[n] = cv2.flip(image, 1).reshape(image.shape)  # np.fliplr(img)  #cv2.flip(img,1) ##left-right
            #print('flip', images[n].shape)
    return images


def random_shift_scale_rotateN(images, shift_limit=(-0.0625, 0.0625), scale_limit=(1 / 1.1, 1.1),
                               rotate_limit=(-45, 45), aspect_limit=(1, 1), borderMode=cv2.BORDER_REFLECT_101, u=0.5):
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height, width, channel = images[0].shape

        angle = random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = random.uniform(scale_limit[0], scale_limit[1])
        aspect = random.uniform(aspect_limit[0], aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = math.cos(angle / 180 * math.pi) * (sx)
        ss = math.sin(angle / 180 * math.pi) * (sy)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        for n, image in enumerate(images):
            images[n] = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                            borderValue=(0, 0,
                                                         0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return images

