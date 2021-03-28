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
        self.labels = labels_rewritten = {  (  0,  0,  0): 19,
                                            (111, 74,  0): 20,
                                            ( 81,  0, 81): 20,
                                            (128, 64,128):   0,
                                            (244, 35,232):   1,
                                            (250,170,160): 20,
                                            (230,150,140): 20,
                                            ( 70, 70, 70):   2,
                                            (102,102,156):   3,
                                            (190,153,153):   4,
                                            (180,165,180): 20,
                                            (150,100,100): 20,
                                            (150,120, 90): 20,
                                            (153,153,153):   5,
                                            (153,153,153): 20,
                                            (250,170, 30):   6,
                                            (220,220,  0):   7,
                                            (107,142, 35):   8,
                                            (152,251,152):   9,
                                            ( 70,130,180):  10,
                                            (220, 20, 60):  11,
                                            (255,  0,  0):  12,
                                            (  0,  0,142):  13,
                                            (  0,  0, 70):  14,
                                            (  0, 60,100):  15,
                                            (  0,  0, 90): 20,
                                            (  0,  0,110): 20,
                                            (  0, 80,100):  16,
                                            (  0,  0,230):  17,
                                            (119, 11, 32):  18,
                                            ( 64,128,128):   2,
                                            (190,153,153):   4,
                                            (150,120, 90): 20,
                                            (153,153,153):   5,
                                            (0,   64, 64): 20,
                                            (0,  128,192): 20,
                                            (128, 64,  0): 20,
                                            (64,  64,128): 20,
                                            (102,  0,  0): 20,
                                            ( 51,  0, 51): 20,
                                            ( 32, 32, 32): 20,
                                                }
    
    def voxelize(self, path, x, y, z):
        cloud = PyntCloud.from_file(path)
        # cloud = PyntCloud.from_instance("open3d", pcd)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=x, n_y=y, n_z=z, regular_bounding_box=False)
        voxelgrid = cloud.structures[voxelgrid_id]
        grid_color = voxelgrid.colors

        x_cords = voxelgrid.voxel_x
        y_cords = voxelgrid.voxel_y
        z_cords = voxelgrid.voxel_z

        voxel = np.zeros((x, y, z)).astype(np.bool)
        color = np.zeros((x, y, z, 3))

        cnt = 0
        for n_x, n_y, n_z in zip(x_cords, y_cords, z_cords):
            voxel[n_x][n_y][n_z] = True
            color[n_x][n_y][n_z] = grid_color[cnt]
            cnt+=1

        color_feat = color.reshape(-1, 3)
        mapped_labels = np.asarray([self.labels[tuple(vec)] for vec in color_feat])
        return mapped_labels.reshape(x, y, z).astype(float)

    def __getitem__(self, index):
        if self.train:
            model_id = self.Y[index]  
        else:
            model_id = self.Y_val[index]    

        model_id_inp = model_id.split("_")[0] + ".dat"

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
            voxel_complete = self.voxelize((os.path.join(self.gt_val, model_id)), 64, 64, 16)     
        return model_id, resample_pcd(partial, 1024), voxel_complete

    def __len__(self):
        return self.len