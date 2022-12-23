# Simple Graphonomy

### Requirements 
```bash
scipy
tensorboardX
numpy
opencv-python
matplotlib
networkx
```

1. Install pytorch
```bash
python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

2. Install requirements
you can install above package by using `python3 -m pip install -r requirements.txt`


3. Inference
```bash
bash inference.sh
```

#
# Citation

```
@inproceedings{Gong2019Graphonomy,
author = {Ke Gong and Yiming Gao and Xiaodan Liang and Xiaohui Shen and Meng Wang and Liang Lin},
title = {Graphonomy: Universal Human Parsing via Graph Transfer Learning},
booktitle = {CVPR},
year = {2019},
}
```

# Contact
if you have any questions about this repo, please feel free to contact 
[gaoym9@mail2.sysu.edu.cn](mailto:gaoym9@mail2.sysu.edu.cn).

##

## Related work
+ Self-supervised Structure-sensitive Learning [SSL](https://github.com/Engineering-Course/LIP_SSL)
+ Joint Body Parsing & Pose Estimation Network  [JPPNet](https://github.com/Engineering-Course/LIP_JPPNet)
+ Instance-level Human Parsing via Part Grouping Network [PGN](https://github.com/Engineering-Course/CIHP_PGN)
+ Graphonomy: Universal Image Parsing via Graph Reasoning and Transfer [paper](https://arxiv.org/abs/2101.10620) [code](https://github.com/Gaoyiminggithub/Graphonomy-Panoptic)
