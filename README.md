CARL-TORCH
==================================
[![DOI](https://zenodo.org/badge/255859123.svg)](https://zenodo.org/badge/latestdoi/255859123)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](http://img.shields.io/badge/arXiv-1506.02169-B31B1B.svg)](https://arxiv.org/abs/1506.02169)

*Work in progress by Leonora Vesterbacka, with input from Stephen Jiggins, Johann Brehmer, Kyle Cranmer and Gilles Louppe*

## Introduction
`carl-torch` is a toolbox for multivariate reweighting using PyTorch. 
`carl-torch` is based on [carl](https://github.com/diana-hep/carl/) originally developed for likelihood ratio estimation, but repurposed to be used as a multivariate reweighting technique to be used in the context of particle physics research. 

## Background
The principle is to reweight one simulation to look like another. 
In the context of particle physics, where a lot of the research is done using simulated data (Monte-Carlos samples), optimizing the use of the generated samples is the name of the game since computing resources are finite. 
Searches for new physics and measurements of known Standard Model processes are relying on Monte-Carlo samples generated with multiple theoretical settings in order to evaluate the effect of these systematic uncertainties. 
As it is computationally impractical to generate samples for all theoretical settings with full statistical power, smaller samples are generated for each variation, and a weight is derived to reweight a nominal sample of full statistical power to look like a sample generated with a variational setting. 

A naive approach to reweighting a sample `p0(x)` to look like another `p1(x)` is by calculating a weight defined as `r(x)=p1(x)/p0(x)` parameterized in one or two dimensions, i.e. as a function of one or two physical variables, and apply this weight to the nominal sample `p0(x)`. 
The obvious drawback of this approach is that one or two dimensions is not nearly enough to capture the effects in the full phase space. 

Therefore, a *multivariate* reweighting technique is proposed which can take into account the full space instead of just two dimensions. 
The technique utilizes a binary classifier trained to differntiate the sample `p0(x)` from sample `p1(x)`, where `x` is an n-dimensional feature vector. 
An ideal classifier will estimate `s(x) = p0(x) / (p0(x) + p1(x))`, and by identifying the weight `r(x) = p1(x) / p0(x)`, the output of the classifier can be rewritten as `s(x) = r(x) / (1 + r(x))`. 
The actual weight `r(x)` is retrieved after expressing `r(x)` as a function of `s(x)`: `r(x) ~ s(x) / (1 - s(x))`. For example three variables (out of n possible ones that can be used in the training) is shown below, for the two distributions `p0(x)` and `p1(x)`. 
<p align="center">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/1_nominalVsVar_1000000.png" width="300">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/2_nominalVsVar_1000000.png" width="300">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/4_nominalVsVar_1000000.png" width="300">
</p>

The classification is done using a PyTorch DNN, with Adam optimizer and relu activation function, and the calibration of the classifier is done using histogram or isotonic regression. Other optimizers and activation functions are available.  
Once the weights have been calculated as a function of the classifier output `r(x) ~ s(x) / (1 - s(x))`, they are applied to the original sample `p0(x)` (orange histogram), and will ideally line up with the `p1(x)` sample (dashed histogram), for the three arbitrary variables (out of n) that was used in the training. 
<p align="center">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/w_1_nominalVsVar_train_1000000_fix.png" width="300">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/w_5_nominalVsVar_train_1000000.png" width="300">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/w_4_nominalVsVar_train_1000000.png" width="300">
</p>

The performance of the weights, i.e. how well the reweighted original sample matches the target one, is assessed by training another classifier to discriminate the original sample with weights applied from a target sample. 
The metric to assess the performance of the weights is the area under the curve (AUC) of the ROC curve of another classifier trained to differentiate the target sample from the nominal sample and from the nominal sample *with* the weights applied. 
If the classifier is able to discriminate between the two samples the resulting AUC is larger than 0.5, as in the case of comparing the original sample `p0(x)` and the target sample `p1(x)`.  
On the other hand, if the weights are applied to the nominal sample, the classifier *is unable to discriminate* the target sample from the weighted original sample, which results in an AUC of exactly 0.5, meaning that the weighted original sample and the target one are *virtually identical!*   
<p align="center">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/roc_nominalVsQSFDOWN_dilepton_train_True.png" width="300">
</p>

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
The [carlAthenaOnnx](https://gitlab.cern.ch/mvesterb/carlathenaonnx/-/tree/master/carlAthenaOnnx) is a package that loads the models trained with carl-torch in AthDerivation production environment.  

## Support
If you have any questions, please email leonora.vesterbacka@cern.ch
