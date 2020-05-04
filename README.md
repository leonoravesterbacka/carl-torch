CARL-TORCH
==================================
*Work in progress by Leonora Vesterbacka, with invaluable input from Johann Brehmer, Kyle Cranmer and Gilles Louppe*

## Introduction
`carl-torch` is a toolbox for density ratio estimation using PyTorch. 
`carl-torch` is based on carl (https://github.com/diana-hep/carl/) originally developed for likelihood ratio estimation, but repurposed to be used as a multivariate reweighting technique to be used in the context of particle physics research. 
## Background
The principle is to reweight one simulation to look like another. 
In the context of particle physics, where a lot of the research is done using simulated data (Monte-Carlos samples), optimizing the use of the generated samples is the name of the game, as computing resources are finite. 
Searches for new physics and measurements of known Standard Model processes are relying on Monte-Carlo samples generated with multiple theoretical settings, in order to evaluate the effect of these systematic uncertainties. 
As it is computationally impractical to generate samples for all theoretical settings with full statistical power, smaller samples are generated for each variation, and a weight is derived to reweight a nominal sample of full statistical power to look like a sample generated with a variational setting. 

A naive approach to reweighting a sample `p0(x)` to look like another `p1(x)` is by calculating a weight defined as `r=p1(x)/p0(x)` parameterized in one or two dimensions, i.e. as a function of one or two physical variables, and apply this weight to the nominal sample `p0(x)`. 
The obvious drawback of this approach is that one or two dimensions is not nearly enough to capture the effects in the full phase space. 

Therefore, a multivariate reweighting technique is proposed, based on density ratio estimation, which can take into account the full space instead of just two dimensions. 
The technique is based on approximating a density ratio `r=p1(x)/p0(x)` as `s(x) / 1 - s(x)`, where `s` is a classifier trained to distinguish samples `x ~ p0` from samples `x ~ p1`, and where `s(x)` is the classifier approximate of the probability `p0(x) / (p0(x) + p1(x))`. 
The classification is done using a PyTorch DNN, with Adam optimizer and sigmoid activation function, and the calibration of the classifier is done using histogram or isotonic regression. 
The performance of the weights, i.e. how well the reweighted original sample matches the target one, is assessed by training a discriminator to differentiate the original distribution with weights applied from a target distribution. 
## Documentation
* Extensive details regarding likelihood-free inference with calibrated
  classifiers can be found in the companion paper _"Approximating Likelihood
  Ratios with Calibrated Discriminative Classifiers", Kyle Cranmer, Juan Pavez,
  Gilles Louppe._
  [http://arxiv.org/abs/1506.02169](http://arxiv.org/abs/1506.02169)

## Installation
The following dependencies are required:
- Numpy
- Scipy
- PyTorch 
- Theano 
- scikit-learn 

Once satisfied, `carl-torch` can be installed from source using the following:
```
git clone https://github.com/leonoravesterbacka/carl-torch.git
```
The code is based on three scripts:
- [train.py](train.py) trains neural networks on loaded data.
- [evaluate.py](evaluate.py) evaluates the neural network by calculating the weights and making validation and ROC plots.
- [calibrate.py](calibrate.py) calibrated network predictions based on histograms of the network output.

This toolbox is adapted from  [MadMiner](http://diana-hep.org/madminer) and [CARL](http://diana-hep.org/carl). 
