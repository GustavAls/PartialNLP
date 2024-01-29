# PartialNLP
This repository contains the source code used for the Thesis

"Partially Stochastic Bayesian Neural Networks for Natural Language Processing"

by Peter Kampen and Gustav Als.

### Requirements

This repo uses the packages [Laplace](https://github.com/GustavAls/laplace-lora) and [Transformers](https://github.com/GustavAls/transformers)

Before running any code the requirements.txt file should be installed.

Additionally, the following two source installations are required:

Laplace:

``
pip install https://github.com/GustavAls/laplace-lora
``

Transformers:

``
pip install https://github.com/GustavAls/transformers
``

### Running the code

In the folder `PartialNLP\hpc` the .bsub files used to run the code on the HPC cluster are located.
Here the configurations used for the experiments can be seen. The files contain a broad range of hyperparameters
used for running the different experiments found in the thesis.