# Restricted Boltzmann Machines - Deep Learning

## Overview
This repository implements a Restricted Boltzmann Machine (RBM) classifier using deep learning techniques. The main functionality includes training the RBM and using it for classification tasks.

## Requirements
To run the main file, ensure you have the following packages installed:
- Python 3.x
- Required libraries (install via `pip install -r requirements.txt`)

## Running the Main File
You can run the main file using the following command:
```bash
python main.py -re <rbm_epoch> -ce <classifier_epoch> -k <k_steps_markov>
```
- `-re` or `--rbm_epoch`: Number of RBM training epochs (default: 3)
- `-ce` or `--classifier_epoch`: Number of classifier training epochs (default: 3)
- `-k` or `--k_steps_markov`: Number of Markov steps (default: 10)

## Hyperparameter Tuning with WandB
To perform hyperparameter tuning using the `wandb_sweep.py` script, you can run:
```bash
python wandb_sweep.py -c <count>
```
- `-c` or `--count`: Number of sweep counts (default: 1)

This will initialize sweeps with the specified count, allowing you to optimize hyperparameters effectively.
