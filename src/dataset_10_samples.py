# import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
#from utils import *
from pyntcloud import PyntCloud

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]
           
class Kitti360(data.Dataset): 
    def __init__(self, cfg, train = True, npoints = 8192):
        if train:
            self.inp = os.path.join(cfg['data']['train_split'], "partial")
            self.gt =  os.path.join(cfg['data']['train_split'], "complete")
            self.X = np.asarray(os.listdir(self.inp))
            self.Y = np.asarray(os.listdir(self.gt))
            self.len = np.shape(self.Y)[0]

        else:
            self.inp_val = os.path.join(cfg['data']['val_split'], "partial") 
            self.gt_val = os.path.join(cfg['data']['val_split'] , "complete")
            self.X_val = np.asarray(os.listdir(self.inp_val))
            self.Y_val = np.asarray(os.listdir(self.gt_val))
            self.len = np.shape(self.Y_val)[0]
        self.npoints = npoints
        self.train = train
        self.pose = '/home/bharadwaj/implementations/DATA/poses.txt'
        self.pose_matrix = np.loadtxt(self.pose)
    
    def voxelize(self, path, x, y, z):
        cloud = PyntCloud.from_file(path)
        # cloud = PyntCloud.from_instance("open3d", pcd)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=x, n_y=y, n_z=z, regular_bounding_box=False)
        voxelgrid = cloud.structures[voxelgrid_id]

        x_cords = voxelgrid.voxel_x
        y_cords = voxelgrid.voxel_y
        z_cords = voxelgrid.voxel_z

        voxel = np.zeros((x, y, z)).astype(np.bool)

        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = True
        return voxel.astype(float)

    def __getitem__(self, index):
        if self.train:
            model_id = self.Y[index]
        else:
            model_id = self.Y_val[index]    

        model_id_inp = model_id.split("_")[0] + ".dat"
        # print(os.path.join(self.inp, model_id))

        def trans_vector(model_id, poses):
            '''
            gets poses from pose.txt for each file
            '''
            id = float(model_id.split('.')[0])
            vec = np.squeeze(poses[poses[:,0] == id])
            reshaped = vec[1:].reshape(3,4)
            return reshaped[:,3:].astype(np.float64)

        def read_pcd(filename, center):
            '''
            reads pcd, converts to float and normalizes
            '''
            point_set = np.loadtxt(filename)
            point_set = point_set - center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            pcd = point_set / dist #scale
            return (torch.from_numpy(np.array(pcd)).float())
        
        center = trans_vector(model_id_inp, self.pose_matrix).transpose()
        if self.train:
            partial =read_pcd(os.path.join(self.inp, model_id_inp), center)
            voxel_complete = self.voxelize((os.path.join(self.gt, model_id)), 64, 64, 16)
        else:
            partial =read_pcd(os.path.join(self.inp_val, model_id_inp), center)
            voxel_complete = self.voxelize((os.path.join(self.gt, model_id)), 64, 64, 16)    
        return model_id, resample_pcd(partial, 1024), voxel_complete

    def __len__(self):
        return self.len