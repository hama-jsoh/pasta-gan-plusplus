import os
import cv2
from utils.resize_image import ResizeImg


def preprocess(img_path: str = 'test_samples/image'):
    fileList = os.listdir(img_path)
    dataRoot = os.path.abspath('.')
    imgList = [os.path.join(dataRoot, img_path, file) for file in fileList]

    for img in imgList:
        filename = img[: img.rfind(".")]
        image = cv2.imread(img)
        resizedImg = ResizeImg(image, 320, 512)
        cv2.imwrite(f"{filename}.jpg", resizedImg)
        os.remove(img)
        print("SUCCESS, The original image was deleted.")


if __name__ == "__main__":
    preprocess()
