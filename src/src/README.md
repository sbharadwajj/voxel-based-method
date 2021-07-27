This entire folder is forked from `https://github.com/autonomousvision/convolutional_occupancy_networks`

#### Changes made:

The file `encoder/pointnet.py` is modefied for our particular setting to fit the following changes:
- voxel grid of shape [64, 64, 16]
- Conv3D as final output layer following 3D U-Net as shown in the image below:

![alt text](https://github.com/sbharadwajj/convONw-baseline2/blob/new-refactor/voxel-skeleton.png?raw=True)