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
           
class Kitti360(data.Dataset): 
    def __init__(self, dataset_path, partial_path, split, pose_path, train = True, weights = False, npoints_partial = 1024, npoints = 2048):
        self.train = train
        self.npoints = npoints
        self.weights = weights
        if self.train:
            self.inp = os.path.join(partial_path, split, "partial")
            self.gt = os.path.join(dataset_path, split, "gt")
            self.X = os.listdir(self.inp)
            self.Y = os.listdir(self.gt)

            # CHOOSE FOR VISUALIZATION
            # sort_x = sorted(self.X) # choose 10
            # self.X = sort_x
            self.len = len(self.X)
            print(self.len)
        else:
            self.inp = os.path.join(partial_path, split, "partial")
            self.gt = os.path.join(dataset_path, split, "gt") #TRAIN FOR NOW
            self.X = os.listdir(self.inp)
            self.Y = os.listdir(self.gt)

            # CHOOSE FOR VISUALIZATION
            # sort_x = sorted(self.X)[0::200] # choose the 100th one
            # self.X = sort_x

            self.len = len(self.X)
            print(self.len)

        # print(self.inp)
        # print(self.gt)
        '''
        loads poses to a dictonary to read
        '''
        self.pose = pose_path
        pose_dict = {}
        poses = os.listdir(self.pose)
        pose_folders = [os.path.join(self.pose, folder) for folder in poses]
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

    def __getitem__(self, index):        
        model_id = self.X[index] 
        
        split_list = model_id.split("_") 
        file_name = split_list[-1].split(".")[0]
        drive_name =  "_".join(split_list[:6])
        center = self.get_translation_vec(file_name, self.pose_dict[drive_name]).transpose()

        partial = self.read_pcd(os.path.join(self.inp, model_id), center)
        complete = np.load(os.path.join(self.gt, model_id))  

        # if self.train:
        #     complete, partial = augment_cloud([complete, partial])          
        return model_id, partial.astype(np.float32), complete

    def __len__(self):
        return self.len