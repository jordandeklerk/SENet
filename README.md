# ResNet on CIFAR-10

<hr>

## Contents

1. [Highlights](#Highlights)
2. [SE ResNet Primer](#ResNet)
3. [Requirements](#Requirements)
4. [Usage](#Usage)
5. [Results](#Results)


<hr>

## Highlights
This project is a implementation from scratch of a slightly modified version of the Squeeze and Excitation ResNet introduced in the paper [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507). We implement this model on the small scale benchmark dataset `CIFAR-10`. One of the goals of this project is to illustrate the speed gain of the ResNet model in comparison to the vision transformer models while maintaining comparable accuracy on `CIFAR-10` and other small-scale datasets. 

<hr>

## Squeeze and Excitation ResNet Primer
Need to complete this

<hr>

## Requirements
```shell
pip install -r requirements.txt
```

<hr>

## Usage
To replicate the reported results, clone this repo
```shell
cd your_directory git clone git@github.com:jordandeklerk/SE-ResNet-pytorch.git
```
and run the main training script
```shell
python train.py 
```

<hr>

## Results
We test our approach on the `CIFAR-10` dataset with the intention to extend our model to 4 other small low resolution datasets: `Tiny-Imagenet`, `CIFAR100`, `CINIC10` and `SVHN`. All training took place on a single A100 GPU.
  * CIFAR10
    * ```SEresnet_cifar10_input32``` - 92.4 @ 32

Flop analysis:
```
total flops: 127105568
total activations: 545849
number of parameter: 861818
| module   | #parameters or shape   | #flops   |
|:---------|:-----------------------|:---------|
| model    | 0.862M                 | 0.127G   |
|  conv1   |  0.432K                |  0.442M  |
|  bn1     |  32                    |  32.768K |
|  layer1  |  42.336K               |  43.205M |
|  layer2  |  0.164M                |  41.805M |
|  layer3  |  0.654M                |  41.616M |
|  fc      |  0.65K                 |  0.64K   |
|  avgpool |                        |  4.096K  |
```
   
<hr>

## Citations
```bibtex
@article{hu2018senet,
  title={Squeeze-and-Excitation Networks},
  author={Jie Hu and Li Shen and Gang Sun},
  journal={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```
