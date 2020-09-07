# GPU HTCondor carl-torch

The following instructions summarise how to operate the carl-torch training package on a GPU. Instructions for operating HTCondor lxplus based GPU can be found here:

https://batchdocs.web.cern.ch/tutorial/exercise10.html

The instructions summarise that to execute on lxplus please follow the following

*   Create a tar package of the necessary scripts and the CARLENV python virtual environment: `$ tar zcvf carl-torch.tar.gz setup.py __init__.py evaluate.py calibrate.py train.py ml/ CARLENV/`
*   Then for an interactive session execute:  `$ condor_submit -interactive train.sub`
*   Once provided a node with a GPU one can see the available GPU information using: `$ nvidia-smi`
*   Unpack the tar package that is pulled into the new node (see train.sub) using: `$ tar -xvf carl-torch.tar.gz`
*   Open the python virtual environment using: `$ source CARLENV/bin/activate`
*   Execute training as normal:  `$ python train.py -s dilepton -v ckkw -n 500000`