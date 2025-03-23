# Training Script Usage Guide

This README provides instructions on how to run the training scripts `train.py` and `train_DA.py`.

## Prerequisites

Before running the scripts, ensure you have the following:

- Python 
- Required libraries (install using `sh env.sh`)

You can run both scripts by:
```
sh run.sh
```

## Running train.py

This script trains the a model on synthetic dataset and tests on real dataset.
To run the `train.py` script, use the following command:

```bash
python train.py 
```


## Running train_DA.

This script applies domain adaptation technique to increase model accuracy on real dataset.
To run the `train_DA.py` script, use the following command:

```bash
python train_DA.py
```
