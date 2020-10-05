CARL-TORCH
==================================
[![DOI](https://zenodo.org/badge/255859123.svg)](https://zenodo.org/badge/latestdoi/255859123)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Work in progress by Leonora Vesterbacka, with input from Stephen Jiggins, Johann Brehmer, Kyle Cranmer and Gilles Louppe*

## Introduction
`carl-torch` is a toolbox for multivariate reweighting using PyTorch. 
`carl-torch` is based on [carl](https://github.com/diana-hep/carl/) originally developed for likelihood ratio estimation, but repurposed to be used as a multivariate reweighting technique to be used in the context of particle physics research. 

## Background
The principle is to reweight one simulation to look like another. 
In the context of particle physics, where a lot of the research is done using simulated data (Monte-Carlos samples), optimizing the use of the generated samples is the name of the game since computing resources are finite. 
Searches for new physics and measurements of known Standard Model processes are relying on Monte-Carlo samples generated with multiple theoretical settings in order to evaluate the effect of these systematic uncertainties. 
As it is computationally impractical to generate samples for all theoretical settings with full statistical power, smaller samples are generated for each variation, and a weight is derived to reweight a nominal sample of full statistical power to look like a sample generated with a variational setting. 

A naive approach to reweighting a sample `p0(x)` to look like another `p1(x)` is by calculating a weight defined as `r=p1(x)/p0(x)` parameterized in one or two dimensions, i.e. as a function of one or two physical variables, and apply this weight to the nominal sample `p0(x)`. 
The obvious drawback of this approach is that one or two dimensions is not nearly enough to capture the effects in the full phase space. 

Therefore, a multivariate reweighting technique is proposed which can take into account the full space instead of just two dimensions. 
The technique is based on approximating a density ratio `r=p1(x)/p0(x)` as `s(x) / 1 - s(x)`, where `s` is a classifier trained to distinguish samples `x ~ p0` from samples `x ~ p1`, and where `s(x)` is the classifier approximate of the probability `p0(x) / (p0(x) + p1(x))`. 
The classification is done using a PyTorch DNN, with Adam optimizer and relu activation function, and the calibration of the classifier is done using histogram or isotonic regression. Other optimizers and activation functions are available.  

The performance of the weights, i.e. how well the reweighted original sample matches the target one, is assessed by training another classifier to discriminate the original distribution with weights applied from a target distribution. 
If the classifier is able to discriminate between the two samples (area under the curve, AUC > 0.5), the weights are not doing a good job, whereas if the classifier is unable to discriminate the target sample from the weighted original sample, the weights are doing a good job (AUC close to 0.5).  

## Documentation
Extensive details regarding likelihood-free inference with calibrated classifiers can be found in th paper _"Approximating Likelihood Ratios with Calibrated Discriminative Classifiers", Kyle Cranmer, Juan Pavez, Gilles Louppe._ [http://arxiv.org/abs/1506.02169](http://arxiv.org/abs/1506.02169)

## Installation
The following dependencies are required:
 - numpy>=1.13.0
 - scipy>=1.0.0
 - scikit-learn>=0.19.0
 - torch>=1.0.0
 - uproot
 - matplotlib>=2.0.0

For hyperparameter search skorch is also required. 

Once satisfied, `carl-torch` can be installed from source using the following:
```
git clone https://github.com/leonoravesterbacka/carl-torch.git
```

## Execution
The code is based on three scripts:
- [train.py](train.py) trains neural networks on loaded data.
- [evaluate.py](evaluate.py) evaluates the neural network by calculating the weights and making validation and ROC plots.
- [calibrate.py](calibrate.py) calibrated network predictions based on histograms of the network output.

Validation plots are made with option plot set to True in [evaluate.py](evaluate.py), and saved in plots/. 

Hyperparameter search for optimization of the classifier is done in branch hyperparameter-search using skorch.

The training is preferrably done on GPUs. [HTCondor_README.md](HTCondor_README.md) includes instructions on how to train on GPUS on HTCondor (ATLAS users only for now). The evaluation and calibration steps are done instantly and thus not require GPUs. 

## Deployment
The model trained in the train.py step is exported to [onnx](https://github.com/onnx/onnx) format to be loaded in a C++ production environment using [onnxruntime](https://github.com/microsoft/onnxruntime). 
#### For ATLAS users
The [carlAthenaOnnx](https://gitlab.cern.ch/mvesterb/carlathenaonnx/-/tree/master/carlAthenaOnnx) is package that loads the models trained with carl-torch in AthDerivation.  

## Support
If you have any questions, please email leonora.vesterbacka@cern.ch
