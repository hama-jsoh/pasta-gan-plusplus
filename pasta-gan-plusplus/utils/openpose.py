import os
import json
import numpy as np
import cv2


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

    def GetKeypoints(self, probMap, threshold=0.1) -> list:
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

    def Inference(self, imgroot: str = "./data/human") -> dict:
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
            image = self._ResizeImg(image, 320, 512)

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
                keypoints = self.GetKeypoints(probMap, threshold)
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

    @staticmethod
    def FileOutput(dict_obj, file_path: str, indent: bool = False) -> None:
        base_path = os.path.abspath(file_path)
        for file_ in dict_obj.keys():
            basename = file_[: file_.rfind('.')]
            filename = basename[basename.rfind('/')+1:]

            jsonfile = f"{filename}_keypoints.json"
            basedir = os.path.join(base_path, 'keypoints')
            filepath = os.path.join(basedir, jsonfile)

            os.makedirs(basedir, exist_ok=True)
            with open(filepath, "w") as j:
                if indent:
                    json.dump(dict_obj[file_], j, ensure_ascii=False, indent=4)
                else:
                    json.dump(dict_obj[file_], j, ensure_ascii=False)

    def _ResizeImg(self, img: np.array, width: int, height: int) -> np.array:
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

    # openpose configuration
    pose = OpenPose(
        model="coco",
        verbose=True,
    )

    # run openpose
    kpts = pose.Inference(imgroot="../test_samples/image")

    # fileio
    pose.FileOutput(
        dict_obj=kpts,
        file_path="./pathtest/",
        indent=False,
    )
