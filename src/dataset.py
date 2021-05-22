# import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
#from utils import *
import pandas as pd
from pyntcloud import PyntCloud

from data_utils import *

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]
           
class Kitti360_previous(data.Dataset): 
    def __init__(self, cfg, train = True, npoints = 8192):
        if train:
            # self.inp = os.path.join(cfg['data']['train_split'], "downsampled_inp/downsampled_train/") # TRAIN ON 100,000 DATA
            self.gt =  os.path.join("/home/bharadwaj/implementations/DATA", "downsampled_fused/downsampled_train/")
            self.inp = cfg['data']['train_split']
            self.X = np.asarray(os.listdir(self.inp))
            self.Y = np.asarray(os.listdir(self.gt))
            self.len = np.shape(self.Y)[0]
        else:
            # self.inp_val = os.path.join(cfg['data']['val_split'], "downsampled_inp/downsampled_predict/") # TRAIN ON 100,000 DATA
            self.inp_val = cfg['data']['val_split']
            self.gt_val = os.path.join("/home/bharadwaj/implementations/DATA" , "downsampled_fused/downsampled_predict/")
            self.X_val = np.asarray(os.listdir(self.inp_val))
            self.Y_val = np.asarray(os.listdir(self.gt_val))
            self.len = np.shape(self.Y_val)[0]
        self.npoints = npoints
        self.train = train
        self.pose = '/home/bharadwaj/implementations/DATA/poses.txt'
        self.pose_matrix = np.loadtxt(self.pose)
    
    def voxelize(self, path, x, y, z):
        point_set = np.loadtxt(path)
        point_dict = {'x':point_set[:,0], 'y':point_set[:,1],'z':point_set[:,2]}
        cloud = PyntCloud(pd.DataFrame(data=point_dict))
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
        
        center = trans_vector(model_id, self.pose_matrix).transpose()
        if self.train:
            partial = read_pcd(os.path.join(self.inp, model_id), center)
            voxel_complete = self.voxelize((os.path.join(self.gt, model_id)), 64, 64, 16)
        else:
            partial =read_pcd(os.path.join(self.inp_val, model_id), center)
            voxel_complete = self.voxelize((os.path.join(self.gt_val, model_id)), 64, 64, 16) 
        return model_id, resample_pcd(partial, 100000), voxel_complete

    def __len__(self):
        return self.len

class Kitti360(data.Dataset): 
    def __init__(self, dataset_path, train = True, weights = False, npoints_partial = 1024, npoints = 2048):
        self.train = train
        self.npoints = npoints
        self.weights = weights
        if self.train:
            self.inp = os.path.join(dataset_path, "train", "partial")
            self.gt = os.path.join(dataset_path, "train", "gt")
            self.X = os.listdir(self.inp)
            self.Y = os.listdir(self.gt)
            # sort_y = sorted(self.Y)[0::2000] # choose 10
            # self.Y = sort_y
            self.len = len(self.Y)
        else:
            self.inp = os.path.join(dataset_path, "val", "partial")
            self.gt = os.path.join(dataset_path, "val", "gt")
            self.X = os.listdir(self.inp)
            self.Y = os.listdir(self.gt)

            sort_y = sorted(self.Y)[0::20] # choose the 100th one
            self.Y = sort_y

            self.len = len(self.Y)

        # print(self.inp)
        # print(self.gt)
        '''
        loads poses to a dictonary to read
        '''
        self.pose = '/home/bharadwaj/dataset/KITTI-360/data_poses'
        pose_dict = {}
        poses = os.listdir(self.pose)
        pose_folders = [os.path.join('/home/bharadwaj/dataset/KITTI-360/data_poses', folder) for folder in poses]
        self.pose_dict = {path.split("/")[-1]:np.loadtxt(path+"/poses.txt") for path in pose_folders}

    def get_translation_vec(self, model_id, poses):
        '''
        gets poses from pose.txt for each file
        '''
        id = float(model_id)
        vec = np.squeeze(poses[poses[:,0] == id])
        reshaped = vec[1:].reshape(3,4)
        return reshaped[:,3:].astype(np.float64)

    def read_pcd(self, filename, center):
        '''
        reads pcd and normalizes
        '''
        point_set = np.load(filename) # .astype(np.float64) saved as float already
        point_set = point_set - center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        pcd = point_set / dist # scale
        return pcd #.astype(np.float)

    def voxelize(self, point_set, x, y, z):
        point_dict = {'x':point_set[:,0], 'y':point_set[:,1],'z':point_set[:,2]}
        cloud = PyntCloud(pd.DataFrame(data=point_dict))
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=x, n_y=y, n_z=z, regular_bounding_box=False)
        voxelgrid = cloud.structures[voxelgrid_id]

        x_cords = voxelgrid.voxel_x
        y_cords = voxelgrid.voxel_y
        z_cords = voxelgrid.voxel_z

        voxel = np.zeros((x, y, z)).astype(np.bool)

        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = True
        return voxel.astype(np.float)

    def __getitem__(self, index):        
        model_id = self.Y[index] 
        
        split_list = model_id.split("_") 
        file_name = split_list[-1].split(".")[0]
        drive_name =  "_".join(split_list[:6])
        center = self.get_translation_vec(file_name, self.pose_dict[drive_name]).transpose()

        partial = self.read_pcd(os.path.join(self.inp, model_id), center)
        complete = self.read_pcd(os.path.join(self.gt, model_id), center)  

        if self.train:
            complete, partial = augment_cloud([complete, partial])          
        return model_id, partial.astype(np.float32), self.voxelize(complete.astype(np.float32), 64, 64, 16)

    def __len__(self):
        return self.len