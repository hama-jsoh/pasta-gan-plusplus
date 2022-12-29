import os
import re
import json
from typing import List, Optional

import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn

from training import dataset as custom_dataset

import legacy
import cv2
import tqdm

import scipy.io as sio
import tqdm

import sys
sys.path.append('./graphonomy')
from graphonomy import Graphonomy


CMAP = sio.loadmat("human_colormap.mat")["colormap"]
CMAP = (CMAP * 256).astype(np.uint8)

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


def num_range(s: str) -> List[int]:
    """Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints."""

    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [int(x) for x in vals]


def generate_images(
    dataroot: str,
    testtxt: str,
    outdir: str,
    testpart: str,
    network_pkl: str = 'checkpoints/pasta-gan++/network-snapshot-004408.pkl',
    noise_mode: str = 'const',
    batchsize: int = 1,
    truncation_psi: float = 1,
    use_sleeve_mask: bool = False,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    os.makedirs(outdir, exist_ok=True)

    if testpart == "full":
        dataset = custom_dataset.UvitonDatasetFull_512_test_full(
            path=dataroot,
            test_txt=testtxt,
            use_sleeve_mask=use_sleeve_mask,
            max_size=None,
            xflip=False,
        )
    elif testpart == "upper":
        dataset = custom_dataset.UvitonDatasetFull_512_test_upper(
            path=dataroot,
            test_txt=testtxt,
            use_sleeve_mask=use_sleeve_mask,
            max_size=None,
            xflip=False,
        )
    elif testpart == "lower":
        dataset = custom_dataset.UvitonDatasetFull_512_test_lower(
            path=dataroot,
            test_txt=testtxt,
            use_sleeve_mask=use_sleeve_mask,
            max_size=None,
            xflip=False,
        )
    else:
        raise ValueError("Invalid value for test part!")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=0
    )

    device = torch.device("cuda")

    for data in tqdm.tqdm(dataloader):
        (
            image,
            clothes,
            pose,
            _,
            norm_img,
            norm_img_lower,
            denorm_upper_clothes,
            denorm_lower_clothes,
            denorm_upper_mask,
            denorm_lower_mask,
            retain_mask,
            skin_average,
            lower_label_map,
            lower_clothes_upper_bound,
            person_name,
            clothes_name,
        ) = data

        image_tensor = image.to(device).to(torch.float32) / 127.5 - 1
        clothes_tensor = clothes.to(device).to(torch.float32) / 127.5 - 1
        pose_tensor = pose.to(device).to(torch.float32) / 127.5 - 1
        norm_img_tensor = norm_img.to(device).to(torch.float32) / 127.5 - 1
        norm_img_lower_tensor = norm_img_lower.to(device).to(torch.float32) / 127.5 - 1

        skin_tensor = skin_average.to(device).to(torch.float32) / 127.5 - 1
        lower_label_map_tensor = (
            lower_label_map.to(device).to(torch.float32) / 127.5 - 1
        )
        lower_clothes_upper_bound_tensor = (
            lower_clothes_upper_bound.to(device).to(torch.float32) / 127.5 - 1
        )

        parts_tensor = torch.cat([norm_img_tensor, norm_img_lower_tensor], dim=1)

        denorm_upper_clothes_tensor = (
            denorm_upper_clothes.to(device).to(torch.float32) / 127.5 - 1
        )
        denorm_upper_mask_tensor = denorm_upper_mask.to(device).to(torch.float32)

        denorm_lower_clothes_tensor = (
            denorm_lower_clothes.to(device).to(torch.float32) / 127.5 - 1
        )
        denorm_lower_mask_tensor = denorm_lower_mask.to(device).to(torch.float32)

        retain_mask_tensor = retain_mask.to(device)
        retain_tensor = image_tensor * retain_mask_tensor - (1 - retain_mask_tensor)
        pose_tensor = torch.cat(
            [pose_tensor, lower_label_map_tensor, lower_clothes_upper_bound_tensor],
            dim=1,
        )
        retain_tensor = torch.cat([retain_tensor, skin_tensor], dim=1)
        gen_z = torch.randn([batchsize, 0], device=device)

        with torch.no_grad():
            gen_c, cat_feat_list = G.style_encoding(parts_tensor, retain_tensor)
            pose_feat = G.const_encoding(pose_tensor)
            ws = G.mapping(gen_z, gen_c)
            cat_feats = {}
            for cat_feat in cat_feat_list:
                h = cat_feat.shape[2]
                cat_feats[str(h)] = cat_feat
            gt_parsing = None
            _, gen_imgs, _ = G.synthesis(
                ws,
                pose_feat,
                cat_feats,
                denorm_upper_clothes_tensor,
                denorm_lower_clothes_tensor,
                denorm_upper_mask_tensor,
                denorm_lower_mask_tensor,
                gt_parsing,
            )

        for ii in range(gen_imgs.size(0)):
            gen_img = gen_imgs[ii].detach().cpu().numpy()
            gen_img = (gen_img.transpose(1, 2, 0) + 1.0) * 127.5
            gen_img = np.clip(gen_img, 0, 255)
            gen_img = gen_img.astype(np.uint8)[..., [2, 1, 0]]

            image_np = image_tensor[ii].detach().cpu().numpy()
            image_np = (image_np.transpose(1, 2, 0) + 1.0) * 127.5
            image_np = image_np.astype(np.uint8)[..., [2, 1, 0]]

            clothes_np = clothes_tensor[ii].detach().cpu().numpy()
            clothes_np = (clothes_np.transpose(1, 2, 0) + 1.0) * 127.5
            clothes_np = clothes_np.astype(np.uint8)[..., [2, 1, 0]]

            result = np.concatenate(
                [
                    gen_img[:, 96:416, :],
                ],
                axis=1,
            )

            person_n = person_name[ii].split("/")[-1]
            clothes_n = clothes_name[ii].split("/")[-1]

            save_name = person_n[:-4] + "___" + clothes_n[:-4] + ".png"
            save_path = os.path.join(outdir, save_name)
            cv2.imwrite(save_path, result)

    print("finish")


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
            model="./graphonomy/data/pretrained_model/inference.pth",
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
                protoFile = "./pretrained_models/pose_deploy_linevec.prototxt"
                weightFile = "./pretrained_models/pose_iter_440000.caffemodel"
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
    # 1. openpose(preprocessing)
    openpose = PreProcessor(
        UriInput("keypoints", "./test_samples/image"),
        FileOutput("json", "./test_samples/keypoints")
    )
    openpose.start()

    # 2. graphonomy(preprocessing)
    graphonomy = PreProcessor(
        UriInput("parsing", "./test_samples/image"),
        FileOutput("img", "./test_samples/parsing")
    )
    graphonomy.start()

    # 3. write_txt(permutation)
    with open("./test_samples/test_pairs.txt", "w") as f:
        filelist = os.listdir("./test_samples/image")
        cloth, human = filelist
        f.write(f"{cloth} {human}")

    # 4. synthesis_result
    generate_images(
        dataroot='test_samples',
        testtxt='test_pairs.txt',
        outdir='test_results/full',
        testpart='full'
    )
