import os
import json
import numpy as np
import cv2

import sys
sys.path.append('../graphonomy')

from inference import Graphonomy


KEYPOINTS_NAME = [
    "Nose",
    "Neck",
    "R-Sho",
    "R-Elb",
    "R-Wr",
    "L-Sho",
    "L-Elb",
    "L-Wr",
    "R-Hip",
    "R-Knee",
    "R-Ank",
    "L-Hip",
    "L-Knee",
    "L-Ank",
    "R-Eye",
    "L-Eye",
    "R-Ear",
    "L-Ear",
]


class UriInput:
    def __init__(
        self,
        uid: str,
        uri: str,
    ) -> None:
        self.dataroot = os.path.abspath(uri)
        self.name = self.__class__.__name__
        self.uid = uid
        self.uri = uri


class FileOutput:
    def __init__(
        self,
        uid: str = "json",
        uri: str = "../test_samples",
        dict_obj = None,
        img_list = None,
        indent: bool = False,
        visualize: bool = False
    ) -> None:
        self.name = self.__class__.__name__
        self.uid = uid
        self.uri = os.path.abspath(uri)
        self._dict_obj = dict_obj
        self._img_list = img_list
        self._indent = indent
        self._visualize = visualize

    def run(self, visualize=False):
        base_path = self.uri
        os.makedirs(base_path, exist_ok=True)

        if self.uid == "json":
            for file_ in self._dict_obj.keys():
                basename = file_[: file_.rfind('.')]
                filename = basename[basename.rfind('/')+1:]

                jsonfile = f"{filename}_keypoints.json"
                filepath = os.path.join(base_path, jsonfile)

                with open(filepath, "w") as j:
                    if self._indent:
                        json.dump(self._dict_obj[file_], j, ensure_ascii=False, indent=4)
                    else:
                        json.dump(self._dict_obj[file_], j, ensure_ascii=False)

        if self.uid == "img":
            #img_list = tqdm(self._img_list)
            for parsing_img, img_path, result in self._img_list:
                basename = img_path[: img_path.rfind('.')]
                filename = basename[basename.rfind('/')+1:]

                imgfile = f"{filename}.png"
                filepath = os.path.join(base_path, imgfile)

                if self._visualize:
                    colormap = f"{filename}_colormap.png"
                    colormap_base = os.path.join(base_path, 'colormap')
                    os.path.makedirs(colormap_base, exist_ok=True)
                    colormap_path = os.path.join(colormap_base, colormap)
                    parsing_img.save(colormap_path)
                cv2.imwrite(filepath, result[0, :, :])


class PreProcessor:
    def __init__(
        self,
        *blocks,
        verbose: bool = True
    ):
        self.verbose = verbose
        self._blocks = {}
        for block in blocks:
            self._blocks[block.name] = block

    def openpose(self):
        pose = OpenPose(
            model="coco",
            verbose=self.verbose
        )
        kpts = pose.run(self._blocks['UriInput'].uri)
        return kpts

    def graphonomy(self):
        humanparse = Graphonomy(
            model="../graphonomy/data/pretrained_model/inference.pth",
            use_gpu=True
        )
        parsing_imgs = humanparse.run(self._blocks['UriInput'].uri)
        return parsing_imgs

    def start(self):
        if self._blocks['UriInput'].uid == "keypoints":
            kpts = self.openpose()
            setattr(self._blocks['FileOutput'], '_dict_obj', kpts)
            self._blocks['FileOutput'].run()
        elif self._blocks['UriInput'].uid == "parsing":
            parsing_imgs = self.graphonomy()
            setattr(self._blocks['FileOutput'], '_img_list', parsing_imgs)
            self._blocks['FileOutput'].run()
        else:
            raise ValueError('Enter Blocks')


class OpenPose:
    def __init__(
        self,
        model,
        verbose: bool = True,
    ) -> None:
        if model is not None:
            if model == "coco":
                protoFile = "../pretrained_models/pose_deploy_linevec.prototxt"
                weightFile = "../pretrained_models/pose_iter_440000.caffemodel"
                self.nPoints = 18
        else:
            raise Exception("model required!, [recommends: 'coco']")
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightFile)

        self.verbose = verbose

    def _get_keypoints(self, probMap, threshold=0.1) -> list:
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []

        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, _, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(
                list(maxLoc + (round(float(probMap[maxLoc[1], maxLoc[0]]), 6),))
            )

        if keypoints:
            keypoints = keypoints[0]
        else:
            keypoints = [0, 0, 0]
        return keypoints

    def run(self, imgroot: str = "./data/human") -> dict:
        dataroot = os.path.abspath(imgroot)
        if self.verbose:
            print("Using CPU device")

        fileList = os.listdir(dataroot)
        imgList = []
        for file in fileList:
            imgPath = os.path.join(dataroot, file)
            imgList.append(imgPath)

        kptList = []
        for img in imgList:
            image = cv2.imread(img)
            image = self._resize_img(image, 320, 512)

            imageHeight, imageWidth, _ = image.shape
            inHeight = 368
            inWidth = int((inHeight / imageHeight) * imageWidth)
            inpBlob = cv2.dnn.blobFromImage(
                image,
                1.0 / 255,
                (inWidth, inHeight),
                (0, 0, 0),
                swapRB=False,
                crop=False,
            )
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            self.net.setInput(inpBlob)
            output = self.net.forward()

            detected_keypoints = []
            threshold = 0.1
            person_id = [-1]

            for part in range(self.nPoints):
                probMap = output[0, part, :, :]
                probMap = cv2.resize(probMap, (image.shape[1], image.shape[0]))
                keypoints = self._get_keypoints(probMap, threshold)
                if self.verbose:
                    print("Keypoints - {} : {}".format(KEYPOINTS_NAME[part], keypoints))
                detected_keypoints.append(keypoints)

            kpts = []
            for kpt in detected_keypoints:
                x, y, p = kpt
                kpts.append(x)
                kpts.append(y)
                kpts.append(p)

            people = [
                dict(
                    person_id=person_id,
                    pose_keypoints_2d=kpts,
                    face_keypoints_2d=[],
                    hand_left_keypoints_2d=[],
                    hand_right_keypoints_2d=[],
                    pose_keypoints_3d=[],
                    face_keypoints_3d=[],
                    hand_left_keypoints_3d=[],
                    hand_right_keypoints_3d=[],
                )
            ]

            jsonForm = dict(version=1.3, people=people)
            kptList.append(jsonForm)
        imgKpt = dict(zip(imgList, kptList))
        return imgKpt

    def _resize_img(self, img: np.array, width: int, height: int) -> np.array:
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
    openpose = PreProcessor(
        UriInput("keypoints", "../test_samples/image"),
        FileOutput("json", "../test_samples2/keypoints")
    )
    openpose.start()

    graphonomy = PreProcessor(
        UriInput("parsing", "../test_samples/image"),
        FileOutput("img", "../test_samples2/parsing")
    )
    graphonomy.start()
