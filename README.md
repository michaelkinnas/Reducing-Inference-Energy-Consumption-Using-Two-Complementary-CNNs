# Reducing the energy requirements of Artificial Νeural Νetworks in Training & Inference

This repository is part of my thesis "Reducing the energy requirements of Artificial Νeural Νetworks in Training & Inference", for Harokopio University of Athens.

It contains three main scripts that run the experiments as described in the paper.

## Main methodology implementation of two heterogeneous CNNs

The file `main.py` will run the main inference workload of the proposed methodology.
To run it you can type `python3 main.py` plus additional parameters as described bellow. A list of supported CNN models is written below.

- -m1 --model1: The first CNN model to use. The selection of the first model will determine which dataset to use, CIFAR-10 or ImageNet
- -m2 --model2: The second CNN model to use. Optional. If not set the script will simply run a single model inference workload of the first model.
- -f --filepath: The directory path of the CIFAR-10 or ImageNet dataset.
- -s --scorefn: The selected of the four score functions to use. Options: (maxp, difference, entropy, oracle).
- -p --postcheck: Enable post-check. Default is false.
- -m --memory: Enable memory component. Default is None. Options: (dhash, invariants).
- -d --duplicates: Set the percentage of the original training set for duplication. Default is 0 (No duplicates). Range (0-1].
- -r --rotations: Set the percentage of the duplicated samples to apply random rotations or flips if a --duplicates value is given. Default is 0. Range (0-1].
- -e --end: What to do when finished. Options: {alarm, shutdown}. Default is shutdown.

### Examples of use

```console
python3 main.py --model1 resnet20 --model2 mobilenetv2_x0_5 --filepath "../data/cifar-10-batches-py" --scorefn difference --threshold 0.8724 --postcheck --end alarm
```

## Find best hyperparameter for max accuracy.

The script file `threshold.py` will calculate the optimal threshold hyperparameter for a given CNN pair. 

To run it use the command `python3 threshold.py` plus some additional parameters as described bellow:

- -m1 --model1: The first CNN model to use.
- -m2 --model2: The second CNN model to use.
- -f --filepath: The directory path of the CIFAR-10 or ImageNet dataset to use.
- -t --train: Only applicable for the CIFAR-10 dataset. Wether to use the training or test dataset.
- -n --n_threshold_values: The number of threshold values between 0 and 1 to check. The greater the number the longer the process. Default is 2000.

### Examples of use

To calculate the best threshold hyperparameter for the selected CNN pair, you can type:

```console
python3 threshold.py --model1 resnet20 --model2 mobilenetv2_x0_5 --filepath "../data/cifar-10-batches-py"
```

_Notice: Normally you would use a validation set for this. However CIFAR-10 does not have a validation set, thus running this script on the test set, will calculate the perfect hypermarameter for best accuracy. In our paper we used the training set although not 100% correct._

## Calculate heterogeneity matrix.

The script file `heterogeneity.py` will calculate the optimal threshold hyperparameter for a given CNN pair. 

To run it use the command `python3 heterogeneity.py` plus some additional parameters as described bellow:

- -d --dataset: Which dataset models to use. Options: {cifar10, imagenet}.
- -f --filepath: The directory path of the CIFAR-10 or ImageNet dataset to use.
- -t --train: Only applicable to the CIFAR-10 dataset. Define whether to use the training or test dataset.

### Examples of use
```console
python3 heterogeneity.py -d cifar10 -f "../data/cifar-10-batches-py"
```
_Notice: Normally you would use a validation set for this. However CIFAR-10 does not have a validation set, thus running this script on the test set, will calculate the perfect hypermarameter for best accuracy. In our paper we used the training set although not 100% correct._

## Supported CNN models

### ImageNet

- convnext_small
- convnext_tiny
- densenet121
- densenet161
- densenet169
- densenet201
- googlenet
- inception_v3
- mnasnet0_5
- mnasnet0_75
- mnasnet1_0
- mnasnet1_3
- mobilenet_v3_small
- mobilenet_v3_large
- regnet_x_16gf
- regnet_x_1_6gf
- regnet_x_3_2gf
- regnet_x_400mf
- regnet_x_800mf
- regnet_x_8gf
- regnet_y_1_6gf
- regnet_y_3_2gf
- regnet_y_400mf
- regnet_y_800mf
- regnet_y_8gf
- resnext50_32x4d
- resnet101
- resnet18
- resnet34
- resnet50
- shufflenet_v2_x0_5
- shufflenet_v2_x1_0
- shufflenet_v2_x1_5
- shufflenet_v2_x2_0
- squeezenet1_0
- squeezenet1_1
- swin_s
- swin_t
- swin_v2_s
- swin_v2_t

### CIFAR-10

- mobilenetv2_x0_5
- mobilenetv2_x0_75
- mobilenetv2_x1_0
- mobilenetv2_x1_4
- repvgg_a0
- repvgg_a1
- repvgg_a2
- resnet20
- resnet32
- resnet44
- resnet56
- shufflenetv2_x0_5
- shufflenetv2_x1_0
- shufflenetv2_x1_5
- shufflenetv2_x2_0
- vgg11_bn
- vgg13_bn
- vgg16_bn
- vgg19_bn


