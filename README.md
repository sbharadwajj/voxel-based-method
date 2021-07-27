# Voxel-Based Methods

This repository contains the code for Voxel-Based Methods split into two main branches:

- main: code for binary voxelization (scene completion)
- semantic: code for semantic scene completion

![alt text](https://github.com/sbharadwajj/convONw-baseline2/blob/new-refactor/voxel-skeleton.png?raw=True)

## Environment

Please use `torch14-cuda10-ConvOccu.recipe` to build a singularity image or use `requirements.txt` to check for dependencies.
It is also necessary to build according to the instructions provided [here](https://github.com/autonomousvision/convolutional_occupancy_networks#installation).

## Train

To train, simply run:
```
singularity exec <path-to-simg> python src/train_baseline.py baseline-num-level-4-18k.yaml
```

Configure `baseline-num-level-4-18k.yaml` and change the necessary paths of dataset and output directory.

The code is borrowed from [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks) and modifications are made.