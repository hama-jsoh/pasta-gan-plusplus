import os
import json
import numpy as np
import cv2


def ResizeImg(img: np.array, width: int, height: int) -> np.array:
    bg_color = img[0][0]
    ratio = width / height
    if img.shape[1] / img.shape[0] < ratio:
        if img.shape[0] >= height:
            w = int(img.shape[1] * height / img.shape[0])
            img = cv2.resize(img, (w, height))
        else:
            w = int(img.shape[1] * height / img.shape[0])
            img = cv2.resize(img, (w, height), interpolation=cv2.INTER_LANCZOS4)
        i = (width - img.shape[1]) // 2
        j = width - img.shape[1] - i
        row = [bg_color for _ in range(height)]
        pr = np.transpose(np.array([row] * i), (1, 0, 2))
        po = np.transpose(np.array([row] * j), (1, 0, 2))
        if i and j:
            img = np.concatenate((pr, img, po), axis=1)
        elif i:
            img = np.concatenate((pr, img), axis=1)
        elif j:
            img = np.concatenate((img, po), axis=1)
        else:
            pass

    else:
        if img.shape[1] >= width:
            h = int(img.shape[0] * width / img.shape[1])
            img = cv2.resize(img, (width, h))
        else:
            h = int(img.shape[0] * width / img.shape[1])
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_LANCZOS4)
        i = (height - img.shape[0]) // 2
        j = height - img.shape[0] - i
        row = [bg_color for _ in range(width)]
        pr = np.array([row] * i)
        po = np.array([row] * j)
        if i and j:
            img = np.concatenate((pr, img, po), axis=0)
        elif i:
            img = np.concatenate((pr, img), axis=0)
        elif j:
            img = np.concatenate((img, po), axis=0)
        else:
            pass
    return img


if __name__ == "__main__":

    dataroot = '/home/ubuntu/projects/eot/PASTA-GAN-plusplus/tests/image'

    fileList = os.listdir(dataroot)
    imgList = []
    for file in fileList:
        if "._human" not in file:
            imgPath = os.path.join(dataroot, file)
            imgList.append(imgPath)
    for img in imgList:
        filename = img[: img.rfind(".")]
        img = cv2.imread(img)
        img = ResizeImg(img, 320, 512)
        cv2.imwrite(f"{filename}.jpg", img)
