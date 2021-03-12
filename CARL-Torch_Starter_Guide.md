# CARL-Torch Instructions

*Calibrated Likelihood Ratio Estimators*, or CARL for short, is a multi-dimensional re-weighting technique that utilises a Neural Network to train a categorical cross-entropy loss function to estimate the likelihood ratio of a data point $\vec{x}$ in an input feature space of $\mathbb{R}^{n}$ dimensions, based on two probability density function models, $\mathcal{P}_{i}(\vec{x}|\vec{\theta}_{i})$ for $i=[1,2]$. Each model is parameterised via the collection of parameters $\theta_{i}^{k}$, where $k$ represents the number of underlying parameters of each probability density function. 

## CARL-Torch

[CARL-Torch](https://github.com/leonoravesterbacka/carl-torch) is [pyTorch](https://pytorch.org/) implementation of a Neural Network that trains on two Monte Carlo models, defined as Model A/B or $\mathcal{P}_{A/B} (\vec{x}|\vec{\theta}_{A/B})$ for short, in order to learn a mapping function between the two models based on the user defined input space ($\mathbb{R}^{n}$). 

To get started there are four key steps:

1.  Clone source code
2.  Setup [Python3 virtual environment](https://docs.python.org/3/library/venv.html) 
3.  Package installations
4.  Execution: Train/Validate/Evaluate Neural Network

### 1) Obtaining Source Code

Please proceed to clone the [GitHub](https://github.com/leonoravesterbacka/carl-torch) repository via the command:

```bash
SSH:
$ git clone git@github.com:leonoravesterbacka/carl-torch.git 

HTTPS:
git clone https://github.com/leonoravesterbacka/carl-torch.git
```

Please note that to clone via ssh the user must setup a ssh-key on your local machine (local computer, remote server, etc...) that is linked to your GitHub account. Instructions on how to do this can be found [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

### 2) Python3 Virtual Envrionment Creation

Python3 has the ability to create [virtual environments](https://docs.python.org/3/library/venv.html) that function as self contained python library installations. These virtual environments allow a user to operate with root-privileges on a host machine with which they do not have host-level root-privileges. 

**NOTE:** This is not needed if the user has root access to the operating system of a machine, as the user can install the necessary python libraries natively on the machine. However, using a Python virtual environment is still good practice in this circumstance regardless as it prevents backwards incompatbility between user projects when updating or expanding Python libraries in the main host machine installation folder. 

To setup a Python environment the user should execute:

```bash
$ python3 -m venv <Path to environment> 
```

Where for instance `<Path to environment> = /path/to/folder/CARLENV` would create in the absolute path location provided a `CARLENV/` folder that would contain the virtual environment. 

### 3) Package Installations

After creating the python virtual environment please proceed to activate the environment using:

```bash
$ source /path/to/folder/CARLENV/bin/activate
```

This will place you inside the `CARLENV` virtual environment. This will be visually obvious by the addition of `(CARLENV)` preceeding your usual UNIX prompt:

```bash
$ source /path/to/folder/CARLENV/bin/activate
(CARLENV) $  
```

Once inside the virtual environment the user can install the necessary packages via either:

*   **Python Package Index:** Using [pip](https://pypi.org/) to install source wheels or source distributions
*   **Conda:** Using [conda](https://conda.io/en/latest/) to install cross-platform libraries via the Anaconda repository as binaries

The packages to install are:

* cachetools==4.1.1
* certifi==2020.6.20
* cycler==0.10.0
* future==0.18.2
* joblib==0.16.0
* kiwisolver==1.2.0
* matplotlib==3.3.1
* numpy==1.19.1
* onnx==1.7.0
* onnxruntime==1.5.2
* pandas==1.1.1
* Pillow==7.2.0
* pyparsing==2.4.7
* python-dateutil==2.8.1
* pytz==2020.1
* scikit-learn==0.23.2
* scipy==1.5.2
* seaborn==0.10.1
* six==1.15.0
* threadpoolctl==2.1.0
* torch==1.6.0
* torchvision==0.7.0
* uproot==3.12.0
* uproot-methods==0.7.4
* uproot4==0.0.27

Which can all be found in the `requirements.txt` file, and installed via conda/pip using:

```bash
PyPi
(CARLENV) $ pip install -r requirements.txt

Conda
(CARLENV) $ conda install --file requirements.txt
```

**NOTE:** Please note that the above `requirements.txt` file was created using PyPi freeze command.

Of course it can happen that the users native Python installations might yield incompatibility issues with the above requirements file. In this instance, one can run the next step of these instruction (Training) and then at each failed import one can execute:

```bash
(CARLENV) $ pip install <import failed package>
```

### 4) Execution

There are three key steps to the [CARL-torch](https://github.com/leonoravesterbacka/carl-torch) package:

1.  Train -> train.py
2.  Evaluate -> evaluate.py
3.  Validate -> validate.py

#### Train

To train the neural network the user is required to have two input root files, one for Model A and one for Model B. To obtain these the [CARLAthenaOnnx](https://gitlab.cern.ch/mvesterb/carlathenaonnx) package can be used inside the ATLAS Athena framework to generate root flat-tuples. 

Once the root tuples containing one branch per training variable are available, place the root files in a directory. For instance, `/eos/user/s/sjiggins/CARLAthena/Tuples-v2/Tuples/`, is created using `mkdir -p <absolute path>`. Inside the directory please label the root files as `Model_A.root` and `Model_B.root`. For example:

```bash
$  mkdir -p <absolte path>/Inputs
$  mv <path to tuple A> <absolte path>/Inputs/Model_A.root
$  mv <path to tuple B> <absolte path>/Inputs/Model_B.root
$  ls <absolute path/Inputs/
Model_A.root  Model_B.root
```

##### Local Running
Setup the Python3 virtual environment (`CARLENV`) and run the `train.py` script:

```bash
$ source <path to venv>/bin/activate
(CARLENV) $ python train.py -s <Identifier for physics> -v <variation> -n <number of events> -p <absolute path>/Inputs/
```

Where the options available to the `train.py` script are:

|  **Option**  |       **Argument**      |          **Comment**            |
|  ---------   |       ------------      |          -----------            |
|      -s      | String identifier to label process etc... | Not critical for operation |
|      -v      | String identifier to label type of MC parameter variation|  Not critical for operation |
|      -n      | Number of events to run over | Default set to 1000 (testing purposes) |
|      -p      | Path to input data | Root tuple as explained above |

##### GPU - HTCondor

Training via GPU is supported on CUDA compatible hardware. For ATLAS users nvidia V100 Tesla Tensor Cores are available on HTCondor via lxplus and thus can be used for training. To utilise the GPU queues on HTCondor the user can find additional resources via the link - [HTCondor nvidia GPUs exercise](https://batchdocs.web.cern.ch/tutorial/exercise10.html). 

To utilise the GPU HTCondor cores the user should use the HTCondor `train.sub` submission file packed into this repository. Specifically, the user should prepare a tarball of the necessary data and source code via:

```bash
$ tar zcvf carl-torch.tar.gz setup.py __init__.py evaluate.py calibrate.py train.py ml/ CARLENV/ data/
```

Where in this instance the `data/` directory is the output from a local execution of the `train.py` script above. This directory stores one `.npy` file per model with `-n` number of events as requested by the user during the above local run step. 

The user should then run in one of two modes:

###### Interactive Mode:

```bash
$ condor_submit -interactive train.sub
```

This will run an interactive session in which the user will be allocated a node with a single GPU from which the user can see the available GPU resouces via `nvidia-smi`. From here unpack the tarball:

```bash
$ tar -vf carl-torch.tar.gz
```

Then proceed as normal via:

```bash
$ source CARLENV/bin/activate
(CARLENV) $ python train.py -s <Identifier for physics> -v <variation> -n <number of events> -p <absolute path>/Inputs/
```

###### Remote Submission

```bash
$ condor_submit train.sub
```
In this instance the script `CARLRun.sh` (bash) will be packaged and executed on the allocated node. As such the user should edit the `CARLRun.sh` script to match the workflow needed.

#### Evaluate

Coming soon!

#### Validate

Coming soon!