from glob import glob

import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import  Path
from skimage.io import imread, imsave, imshow


def show_graph(img, pred, mask):
    imgs = [img, pred, mask]
    show_graphs(imgs)


def show_graphs(imgs, titles=None):
    show_graph_with_col(imgs, max_cols=4, titles=titles)


def show_graph_with_col(imgs, max_cols, titles=None):
    length = len(imgs)
    if length < max_cols:
        max_cols = length

    max_line = np.ceil(length / max_cols)
    for i in range(1, length+1):
        ax = plt.subplot(max_line, max_cols, i)
        plt.imshow(imgs[i-1], cmap="gray")
        if titles is not None:
            ax.set_title(titles[i-1])
    plt.show()

def show_contour(img):
    a = cv2.imread(img)
    img = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    image, cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(1, len(cnts[0])):
    #     point1 = cnts[0][i - 1][0]
    #     point2 = cnts[0][i][0]
    #     cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), (0, 255, 0), 2)
    # point1 = cnts[0][len(cnts[0]) - 1][0]
    # point2 = cnts[0][0][0]
    # cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), 115, 5)

    contour = cv2.drawContours(img, cnts, 0, 200, 1)
    # imshow(img)
    plt.imshow(contour)
    plt.show()

def show_output_imgs():
    src_path = '/data/zj/data/small_test'
    target_path = '/data/zj/data/small_test_1'
    src_imgs = np.array(glob(src_path + '/**/*.jpg', recursive=True))
    s1 = "criterion_"
    for img_file in src_imgs:
        img_path = Path(img_file)
        img_path_list = glob(target_path + "/" + img_path.stem + "**.*")
        img_path_list.sort()
        titles = []
        images = []
        for i in range(len(img_path_list)):
            images.append(imread(img_path_list[i]))
            title = Path(img_path_list[i]).stem.replace(img_path.stem, "").replace(s1, "")
            titles.append(title)
        show_graphs(images, titles)


if __name__ == '__main__':
    # imgs = np.array(glob('/data/zj/data/small_test/**/*.*', recursive=True))
    # images = []
    # for i in range(9):
    #     images.append(imread(imgs[i]))
    # show_graphs(np.array(images))
    show_output_imgs()

