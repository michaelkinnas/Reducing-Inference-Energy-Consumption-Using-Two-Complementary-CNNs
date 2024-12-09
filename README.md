# Reducing inference energy consumption using dual complementary CNNs

This repository contains the PyTorch implementation of the paper "Reducing inference energy consumption using dual complementary CNNs", published in the FGCS journal. You can read it [here](https://doi.org/10.1016/j.future.2024.107606).

<img src="https://github.com/michaelkinnas/Reducing-Inference-Energy-Consumption-Using-Two-Complementary-CNNs/blob/main/overview.png?raw=true" width="100%">

It contains three scripts that run the experiments as described in the paper, and are explained bellow.

## Prerequisites

- Python 3
- Pytorch >= 2.1.0
- Torchvision >= 0.16.0

## Preparation

To start with the experiments you begin by cloning the repository:

```bash
git clone https://github.com/michaelkinnas/Reducing-Inference-Energy-Consumption-Using-Two-Complementary-CNNs
cd Reducing-Inference-Energy-Consumption-Using-Two-Complementary-CNNs
```

To install the depedencies run:

```bash
pip install -r requirements.txt
```

It is recommended to use a python virtual environment:

- with python venv:

```bash
python -m venv <path-to-environment>/<environment-name>
source  <path-to-environment>/<environment-name>/bin/activate
pip install -r requirements.txt
```

- with conda:

```bash
conda create --name <environment-name> --file requirements.txt
```

## Recommended pipeline

The pipeline as described in the paper has three main steps:

1. Calculate complementarity matrix from given pretrained CNN models and select two CNN models.
2. Calculate λ hyperparamter value for selected CNN pair.
3. Run the inference methodology on test dataset with the selected CNN pair and calculated λ parameter.

## Script 1: Calculate complementarity matrix.

The script file `complementarity.py` will calculate a complementarity matrix for a given pretrained CNN model pool.

To run it use the command `python3 complementarity.py` plus some additional parameters as described bellow:

```
  -h, --help            show this help message and exit
  -D, --dataset {cifar10,imagenet,intel,fashionmnist}
                        Define which dataset models to use.
  -f, --dataset-root DATASET_ROOT
                        The root file path of the validation or test dataset. (e.g. For CIFAR-10 the directory
                        containing the 'cifar-10-batches-py' folder, etc.)
  -t, --train           Define whether to use the training or test split, for datasets that require that
                        parameter.
  -w, --weights WEIGHTS
                        Optional. The path directory of custom weights for all the models used in the process.
                        The files should be in '.pth' extension and named after the original CIFAR-10 model name
                        (e.g. 'resnet20.pth'). If not set the default pretrained CIFAR-10 model weights will be
                        used.
```

The results are saved in a `complementarity.csv` file in the same directory that you run the script.

### Examples of use
```console
python3 complementarity.py -D cifar10 -f "<path to dataset root>"
```

<!-- #### NOTICE: If using Intel of FashionMNIST datasets you must provide the directory that contain all the model weights for all corresponding CIFAR-10 models as shown bellow. The model weights must have the same name as the model, with the .pth extention. If not provided the default CIFAR-10 weights will be used. -->

#### NOTICE: If using the Intel or FashionMNIST datasets you must provide the directory that contain all the .pth model weights for all corresponding CIFAR-10 models as shown bellow. (You can download the .tar.gz files containing the pretrained weights for Intel [here](https://drive.google.com/file/d/1C0cri5ek32RsIujW5GZoWxrcg1pgBWUc/view?usp=drive_link) and for FashionMNIST [here](https://drive.google.com/file/d/1ldod2qrkYISxyIBjA7Rl873aRgdGQnH2/view?usp=drive_link), and then unzip them in that directory), otherwise the default CIFAR-10 weights from PyTorch will be used.

## Script 2: Calculate thresold hyperparameter 'λ'.

The script file `threshold.py` will calculate the optimal threshold hyperparameter for a given CNN pair. 

To run it use the command `python3 threshold.py` plus some additional parameters as described bellow:

```
  -h, --help            show this help message and exit
  -D, --dataset {cifar10,imagenet,intel,fashionmnist}
                        Define which dataset models to use.
  -f, --dataset-root DATASET_ROOT
                        The root file path of the validation or test dataset. (e.g. For CIFAR-10 the directory
                        containing the 'cifar-10-batches-py' folder, etc.)
  -m1, --model1 MODEL1  The first model, required.
  -m2, --model2 MODEL2  The second model, required.
  -t, --train           Only valid for the CIFAR-10 dataset. Define wether to use the training or test dataset.
  -n, --n_threshold_values N_THRESHOLD_VALUES
                        Define the number of threshold values to check between 0 and 1. Higher numbers will be
                        slower. Default is 2000
  -w1, --weights1 WEIGHTS1
                        Optional. Directory of the '.pth' weights file for the first model.
  -w2, --weights2 WEIGHTS2
                        Optional. Directory of the '.pth' weights file for the second model.
```

### Examples of use

To calculate the best threshold hyperparameter for the selected CNN pair, you can type:

```console
python3 threshold.py --model1 resnet20 --model2 mobilenetv2_x0_5 --filepath "<path to dataset root>" -t
```

#### NOTICE: If using Intel of FashionMNIST datasets you must provide the weights .pth file for each model that is trained on this dataset. By default the pretrained model weights are for the CIFAR-10 dataset. You can download the .tar.gz files containing all the pretrained weights for Intel [here](https://drive.google.com/file/d/1C0cri5ek32RsIujW5GZoWxrcg1pgBWUc/view?usp=drive_link) and for FashionMNIST [here](https://drive.google.com/file/d/1ldod2qrkYISxyIBjA7Rl873aRgdGQnH2/view?usp=drive_link). You can unzip the .tar.gz files and point to the corresponding CIFAR-10 model you are using.


## Script 3: Main methodology implementation of two complementary CNNs

```
  -h, --help            show this help message and exit
  -y, --yml-file YML_FILE
                        Use .yml configuration file instead of cli arguments. In this case you must provide the
                        location of the .yml file and the rest of the arguments are ignored.
  -D, --dataset {cifar10,imagenet,intel,fashionmnist}
                        The dataset to use.
  -m1, --model-1 MODEL_1
                        The first model name, required. It must be included in the provided lists of available
                        models.
  -m2, --model-2 MODEL_2
                        The second model. It must be included in the provided lists of available models.
  -w1, --weights-1 WEIGHTS_1
                        Optional. A file path to the first model's weights file.
  -w2, --weights-2 WEIGHTS_2
                        Optional. A file path to the second model's weights file.
  -f, --dataset-root DATASET_ROOT
                        The root file path of the validation or test dataset. (e.g. For CIFAR-10 the directory
                        containing the 'cifar-10-batches-py' folder, etc.)
  -s, --scorefn {maxp,difference,entropy,truth}
                        Score function to use.
  -t, --threshold THRESHOLD
                        The threshold value to use for the threshold check. (λ parameter)
  -p, --postcheck       Enable post check. Default is false.
  -m, --memory {dhash,invariants}
                        Enable memory component. Default is None.
  -d, --duplicates DUPLICATES
                        Set the percentage of the original training set for duplication. Default is 0 (No
                        duplicates). Range [0-1]
  -r, --rotations       If set the duplicated samples will be randomly rotated or mirrored.
  -rp, --root-password ROOT_PASSWORD
                        Optional. If provided the password will be used to command the computer to shutdown after finishing.
```

### Examples of use

```console
python3 main.py --model1 resnet20 --model2 mobilenetv2_x0_5 --filepath "<path to dataset root>" --scorefn difference --threshold 0.8724 --postcheck
```

Instead of cli parameters you can use a configuration file. If you use the `-y` parameter you must provide the `*.yml` file to read from. An example yml configuration file is provided in this repo.


#### NOTICE: If using Intel of FashionMNIST datasets you must provide the weights .pth file for each model that is trained on this dataset. By default the pretrained model weights are for the CIFAR-10 dataset. You can download the .tar.gz files containing all the pretrained weights for Intel [here](https://drive.google.com/file/d/1C0cri5ek32RsIujW5GZoWxrcg1pgBWUc/view?usp=drive_link) and for FashionMNIST [here](https://drive.google.com/file/d/1ldod2qrkYISxyIBjA7Rl873aRgdGQnH2/view?usp=drive_link). You can unzip the .tar.gz files and point to the corresponding CIFAR-10 model you are using.

## Supported CNN models

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

### ImageNet

- convnext_tiny
- densenet121
- densenet161
- densenet169
- densenet201
- googlenet
- inception_v3
- mnasnet1_0
- mnasnet1_3
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
- resnet34
- resnet50
- shufflenet_v2_x2_0
- swin_s
- swin_t
- swin_v2_s
- swin_v2_t

## License

This project is licensed under the Apache License 2.0. You can red the [LICENSE](https://github.com/michaelkinnas/Reducing-Inference-Energy-Consumption-Using-Two-Complementary-CNNs/blob/main/LICENSE) file for more details.