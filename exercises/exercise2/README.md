# Exercise 2: First Person Action Recognition (FPAR)

## Objective
1. Train a Convolutional LSTM for First Person Action Recognition (FPAR):
- Network: ResNet 34 + RNN
- Videos: GTEA61
2. Exercise Steps:
- Learning without Temporal Information (Avgpool)
- Learning with Temporal Information (LSTM)
- Learning with Spatio-Temporal Information (ConvLSTM)

## Structure
- `README.md`: The file with the exercise description.
- `main.py`: The main file to run the exercise.
- `src/`: The folder with the source code.
  - `spatial_transforms.py`: The file with the spatial transformations taken from PyTorch source code.
  - `resnetMod.py`: The file with the ResNet 34 model taken from PyTorch source code.
  - `utils.py`: Loggers and other utilities.
  - `gtea_dataset.py`: The file with the GTEA61 dataset class.

## Dataset
The GTEA61 dataset is available at: [GTEA61](http://cbs.ic.gatech.edu/fpv/)
- synROD: Synthetic dataset with 61 action classes.