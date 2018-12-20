# SafeAI: Predictive Uncertainty DL Models
[![EpiSci](https://img.shields.io/badge/Episys-Science-5fa9d3.svg)](http://www.episci.com/)
[![CircleCI](https://circleci.com/gh/EpiSci/SafeAI.svg?style=svg)](https://circleci.com/gh/EpiSci/SafeAI)
[![Maintainability](https://api.codeclimate.com/v1/badges/2d74bd6e1afde4373ddb/maintainability)](https://codeclimate.com/github/EpiSci/SafeAI/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/2d74bd6e1afde4373ddb/test_coverage)](https://codeclimate.com/github/EpiSci/SafeAI/test_coverage)  
[![python](https://img.shields.io/badge/python-3.6_|2.7-blue.svg)](https://www.python.org/)
[![tensorflow](https://img.shields.io/badge/tensorflow-1.10-ed6c20.svg)](https://www.tensorflow.org/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/safeai.svg)](https://pypi.python.org/pypi/safeai/)
![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)

<img src=./assets/SafeguardAI-logo.png width="490px" height="210px">  

Reusable, easy-to-use tensorflow uncertainty module package.  
Under active development.

## Predictive Uncertainty in Deep Learning Models:
Considering deep learning workflow, uncertainty plays a critical role when applying in real
applications and is essential property in a safety issue. Autonomous driving,
robotics, or critical decision-making, for instance, such domains are especially
caring for safety which is uncertainty in deep learning.  

Recently, plenty of researches related to uncertainty is actively ongoing.
In this project, we leverage TensorFlow to additionally model uncertainty in standard
deep learning architectures. Essentially, we focus on reusability and easy-to-use that
only least numbers of parameters are necessary to run target functions.


## Installation
```bash
# Install using pip
$ pip install safeai

# Or, install from source:
$ git clone https://github.com/EpiSci/SafeAI &&\
$ cd SafeAI && pip install -e .
```

#### Tensorflow, Python version
Currently, SafeAI is being developed and tested with tensorflow version **1.10**,
under both python version **3.6** and **2.7**.  

## SafeAI models
SafeAI models are implemented -- [introducing overall concept of our code.]
```python
import tensorflow as tf
from safeai.models import confident_classifier

MESSAGE = "And Short, intuitive sample code goes here"
```

## Run other examples & test code
Every python code in SafeAI was not meant to be directly run as a single script.
Please have them executed as a module, with [-m flag](https://docs.python.org/3.7/using/cmdline.html#cmdoption-m) 
for testing and running the examples in the project.  
Also **they need to be run from project root folder**, not at `examples/` directory, nor `safeai/tests/`.
```bash
# Clone project, cd into project, install dependencies
$ git clone https://github.com/EpiSci/SafeAI &&\
$ cd SafeAI && pip install -e .

# To run the example:
$ python -m examples.[script name without '.py'] # e.g.) python -m examples.joint_confident

# To execute all tests
$ python -m unittest discover
```
## List of predictive uncertainty models available:
- **Joint Confident Classifier** [(safeai/models/joint_confident.py)](https://github.com/EpiSci/SafeAI/blob/master/safeai/models/joint_confident/joint_confident.py)  
: *Training Confidence-Calibrated Classifier for Detecting Out-of-distribution Samples(2017)*  
Kimin Lee et al | [arxiv.org/abs](https://arxiv.org/abs/1711.09325) | [Author's repository](https://github.com/alinlab/Confident_classifier)  

- List other models in the same form as above

## Contribute to project:
We appreciate your interest in this project!  
Before making an issue, please consider reading [Our contribution guideline.](https://github.com/EpiSci/SafeAI/blob/master/CONTRIBUTING.md)
