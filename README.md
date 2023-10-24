# CrackPropNet

## Introduction

CrackPropNet is an optical flow-based deep neural network for crack propagation measurement in asphalt concrete cracking tests such as semi-circular bending (SCB) test and indirect tension asphalt cracking test (IDEAL-CT). Simply paint your specimen surface with a random black pattern on top of a thin layer of white paint ([tutorial](https://www.youtube.com/)); record a video of the specimen surface during testing ([tutorial](https://www.youtube.com/)). Crack propagation can be accurately and efficiently retrived. 

[*Read More*](https://www.tandfonline.com/doi/full/10.1080/10298436.2023.2186407) ([*Alternative Link*](Zhu_Al_Qadi_2023.pdf))

## Installation

```
git clone https://github.com/zehuiz2/CrackPropNet.git
cd CrackPropNet

# install custom layers
bash install.sh
```

## Python requirements

Currently, the code supports Python 3.8
* numpy
* Pillow
* torch==1.7.1
* torchvision==0.2.2
* tqdm
* glob


## Custom layers

CrackPropNet achitecture relies on custom layers Resample2d or Correlation.
A pytorch implementation of these layers with cuda kernels are available at ./networks.

## Pretrained weights

The pretrained weights of CrackPropNet is available [*here*](https://drive.google.com/file/d/12-ARk1DRcm1B-Uv0g8HnjXrhzZQJPhZO/view?usp=drive_link).

## Example

Sample input images are available at ./img. Reference image should always end with `_0.png`. Deformed images should end with `_*.png`. Sample outputs are available at ./output.

## Reference

If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:

````
@article{author = {Zehui Zhu and Imad L. Al-Qadi},
title = {Automated crack propagation measurement on asphalt concrete specimens using an optical flow-based deep neural network},
journal = {International Journal of Pavement Engineering},
volume = {24},
number = {1},
pages = {2186407},
year  = {2023},
publisher = {Taylor & Francis},
doi = {10.1080/10298436.2023.2186407}
}
````

## Acknowledgement

This implementation is based on the Pytorch implmentation of FlowNetCSS from [*FlowNet 2.0*](https://github.com/NVIDIA/flownet2-pytorch)
