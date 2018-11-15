import numpy as np
import pydicom
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def hist_img(img, minimun_count = 0):
    pixel_count = []
    for i in range(0, 256):
        pixel_count.append((img == i).sum())
    count = []
    idx = []
    for i in range(len(pixel_count)):
        if pixel_count[i] != 0 and pixel_count[i] > minimun_count:
            count.append(pixel_count[i])
            idx.append(i)
    return idx, count

def relocate_pixel(img, threshold=10):
    idx, count = hist_img(img, minimun_count=10)
    length = len(idx)
    if length < threshold:
        print("relocate")
        step = 255 // length
        for i in range(len(idx)):
            img[img==idx[i]] = step * i
    return img

def histogram_equalization(img):
    idx, count = hist_img(img)
    count = np.array(count)
    count_prob = count / count.sum()
    for i in range(1, len(count_prob)):
        count_prob[i] += count_prob[i-1]
    assert count_prob[-1] == 1

    for i in range(len(idx)):
        img[img==idx[i]] = 255 * count_prob[i]
    return img

def analyze_img(img):
    idx, count = hist_img(img, minimun_count=20)
    sns.barplot(idx, count)
    plt.show()

dark_path = 'ct_data/19/imgs/020.jpg'
img = plt.imread(dark_path)
img.flags.writeable = True

def getNumpyData(filename):
    img = pydicom.dcmread(filename)
    return img.pixel_array

def to_gray_scale(img):
    return img.dot([[[0.114, 0.587, 0.299]]])

if __name__ == '__main__':
    img = to_gray_scale(getNumpyData('ct_data/19/DICOM_anon/i0020,0000b.dcm'))
    analyze_img(img)
    #
    plt.imshow(img)
    plt.show()

    plt.imshow(histogram_equalization(img))
    plt.show()

    plt.imshow(relocate_pixel(img))
    plt.show()