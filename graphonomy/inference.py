import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2

from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

import argparse
import torch.nn.functional as F


label_colours = [
    (0, 0, 0),
    (128, 0, 0),
    (255, 0, 0),
    (0, 85, 0),
    (170, 0, 51),
    (255, 85, 0),
    (0, 0, 85),
    (0, 119, 221),
    (85, 85, 0),
    (0, 85, 85),
    (85, 51, 0),
    (52, 86, 128),
    (0, 128, 0),
    (0, 0, 255),
    (51, 170, 221),
    (0, 255, 255),
    (85, 255, 170),
    (170, 255, 85),
    (255, 255, 0),
    (255, 170, 0),
]


def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadmodel", default="./data/pretrained_model/inference.pth", type=str)
    parser.add_argument("--dataroot", default="./img/", type=str)
    parser.add_argument("--output_path", default="./img/parsing/", type=str)
    parser.add_argument("--use_gpu", default=True, type=bool)
    parser.add_argument("--visualize", default=False, type=bool)
    args = parser.parse_args()
    return args


ARGS = set_arguments()
MODEL = ARGS.loadmodel
DATAROOT = ARGS.dataroot
OUTPUT_PATH = ARGS.output_path
USE_GPU = ARGS.use_gpu
VISUALIZE = ARGS.visualize


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


def flip_cihp(tail_list):
    """
    Args:
      tail_list: tail_list size is 1 x n_class x h x w

    Returns:
      torch.Tensor
    """
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev, dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert n >= num_images, (
        "Batch size %d should be greater or equal than number of images to save %d."
        % (n, num_images)
    )
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new("RGB", (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def read_img(img_path):
    _img = Image.open(img_path).convert("RGB")  # return is RGB pic
    return _img


def img_transform(img, transform=None):
    sample = {"image": img, "label": 0}

    sample = transform(sample)
    return sample


def inference(net, dataroot: str, use_gpu=True):
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = (
        adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)
    )

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]

    fileList = os.listdir(dataroot)
    imgList = []
    for file in fileList:
        imgPath = os.path.join(dataroot, file)
        imgList.append(imgPath)

    pimgList = []
    resultList = []
    for image in imgList:
        img = read_img(image)
        testloader_list = []
        testloader_flip_list = []
        for pv in scale_list:
            composed_transforms_ts = transforms.Compose(
                [
                    tr.Scale_only_img(pv),
                    tr.Normalize_xception_tf_only_img(),
                    tr.ToTensor_only_img(),
                ]
            )

            composed_transforms_ts_flip = transforms.Compose(
                [
                    tr.Scale_only_img(pv),
                    tr.HorizontalFlip_only_img(),
                    tr.Normalize_xception_tf_only_img(),
                    tr.ToTensor_only_img(),
                ]
            )

            testloader_list.append(img_transform(img, composed_transforms_ts))
            testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))

        start_time = timeit.default_timer()
        net.eval()

        for iii, sample_batched in enumerate(
            zip(testloader_list, testloader_flip_list)
        ):
            inputs, labels = sample_batched[0]["image"], sample_batched[0]["label"]
            inputs_f, _ = sample_batched[1]["image"], sample_batched[1]["label"]
            inputs = inputs.unsqueeze(0)
            inputs_f = inputs_f.unsqueeze(0)
            inputs = torch.cat((inputs, inputs_f), dim=0)
            if iii == 0:
                _, _, h, w = inputs.size()

            # Forward pass of the mini-batch
            inputs = Variable(inputs, requires_grad=False)

            with torch.no_grad():
                if use_gpu >= 0:
                    inputs = inputs.cuda()
                outputs = net.forward(
                    inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda()
                )
                outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii > 0:
                    outputs = F.upsample(
                        outputs, size=(h, w), mode="bilinear", align_corners=True
                    )
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()

        predictions = torch.max(outputs_final, 1)[1]
        results = predictions.cpu().numpy()
        resultList.append(results)
        vis_res = decode_labels(results)

        parsing_im = Image.fromarray(vis_res[0])
        pimgList.append(parsing_im)
    results = list(zip(pimgList, imgList, resultList))
    return results


class Graphonomy:
    def __init__(
        self,
        model: str,
        use_gpu: bool = True,
        visualize: bool = True
    ) -> None:
        self.net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
            n_classes=20, hidden_layers=128, source_classes=7,
        )
        self.model = model
        self.visualize = visualize
        if self.model is not None:
            x = torch.load(self.model)
            self.net.load_source_model(x)
            print(f"load model: {self.model}")
        else:
            print(f"no model load!!")
            raise RuntimeError("No model!!")
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.net.cuda()
        else:
            raise RuntimeError("must use the gpu!!")

    def run(self, dataroot: str):
        dataroot = os.path.abspath(dataroot)
        parsing_list = inference(
            net=self.net,
            dataroot=dataroot,
            use_gpu=self.use_gpu
        )
        return parsing_list

    def fileoutput(self, pimgs, output_path):
        os.makedirs(output_path, exist_ok=True)
        base_path = os.path.abspath(output_path)
        pimgs = tqdm(pimgs)
        for parsing_img, img_path, result in pimgs:
            basename = img_path[: img_path.rfind('.')]
            filename = basename[basename.rfind('/')+1:]

            imgfile = f"{filename}.png"
            filepath = os.path.join(base_path, imgfile)

            if self.visualize:
                colormap = f"{filename}_colormap.png"
                colormap_path = os.path.join(base_path, colormap)
                parsing_img.save(colormap_path)
            cv2.imwrite(filepath, result[0, :, :])


if __name__ == "__main__":
    graphonomy = Graphonomy(
        model=MODEL,
        use_gpu=USE_GPU,
        visualize=VISUALIZE
    )
    parsing_imgs = graphonomy.run(dataroot=DATAROOT)
    graphonomy.fileoutput(parsing_imgs, output_path=OUTPUT_PATH)
