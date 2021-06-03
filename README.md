# Activation-Sharing-with-Asymmetric-Paths

This repository is the official implementation of Activation-Sharing-with-Asymmetirc-Paths. 

+ The proposed biologically plausible algorithm supports training deep convolutional networks without the weight transport problem and bidirectional connections
+ The proposed biologically plausible algorithm significantly cand reduce memory access overhead when implemented in hardware.

<p align="center"><img src="./Fig/ASAP.png"  width="500" height="500">

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --dataset <type of dataset> --model <type of model> --feedback <type of feedback> --augmentation
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python train.py --dataset <type of dataset> --model <type of model> --feedback <type of feedback> --augmentation
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on MNSIT, SVHN, CIFAR-10, CIFAR-100. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

<p align="center"><img src="./Fig/table of result.PNG"  width="500" height="500">
<p align="center"><img src="./Fig/graph of result.PNG"  width="500" height="500">

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
