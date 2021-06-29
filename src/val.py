import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
from torchsummary import summary
from dataset import *
import math
from src.utils.libkdtree import KDTree
from src.utils.libmesh import check_mesh_contains
# # from pytorch3d.loss import chamfer_distance
# from chamfer_torch import chamfer_distance_reduce
def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def eval_pointcloud(pointcloud, pointcloud_tgt,
                    normals=None, normals_tgt=None,
                    thresholds=np.linspace(1./1000, 1, 1000)):
    ''' Evaluates a point cloud.

    Args:
        pointcloud (numpy array): predicted point cloud
        pointcloud_tgt (numpy array): target point cloud
        normals (numpy array): predicted normals
        normals_tgt (numpy array): target normals
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    # Return maximum losses if pointcloud is empty
    if pointcloud.shape[0] == 0:
        logger.warn('Empty pointcloud / mesh detected!')
        out_dict = EMPTY_PCL_DICT.copy()
        if normals is not None and normals_tgt is not None:
            out_dict.update(EMPTY_PCL_DICT_NORMALS)
        return out_dict

    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud, normals
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud, normals, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]

    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer-L2': chamferL2,
        'chamfer-L1': chamferL1,
        'f-score': F[9], # threshold = 1.0%
        'f-score-15': F[14], # threshold = 1.5%
        'f-score-20': F[19], # threshold = 2.0%
    }

    return out_dict

def pred_to_labels(logits):
    sm = torch.nn.Softmax(dim=1)
    softmax_pred = sm(logits).cpu().numpy()
    pred = np.argmax(softmax_pred, axis=1).astype(np.float)
    return pred

def chamfer_distance_voxel_pcd(pred, gt):
        mask_arr = pred
        mask_arr[pred != 19] = 1.0
        mask_arr[pred == 19] = 0.0
        np_points = (np.vstack(np.where(mask_arr==1))).transpose()
        
        mask_arr_gt = gt
        mask_arr_gt[mask_arr_gt != 19] = 1.0
        mask_arr_gt[mask_arr_gt == 19] = 0.0
        np_points_gt = (np.vstack(np.where(mask_arr_gt==1))).transpose()
        out_dict = eval_pointcloud(np_points_gt, np_points)
        return out_dict["chamfer-L1"]

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_dataset = Kitti360(dataset_path="/home/bharadwaj/dataset/scripts/4096-8192-kitti360-semantic/", train=True, weights=False , npoints_partial = 4096, npoints=8192)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                        shuffle=True, num_workers=8, drop_last=True)
val_dataset = Kitti360("/home/bharadwaj/dataset/scripts/4096-8192-kitti360-semantic/", train=False, weights=False, npoints_partial = 4096, npoints=8192)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8,
                                        shuffle=False, num_workers=8, drop_last=True)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Generator
generator = config.get_generator(model, cfg, device=device)

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('120model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

val_avg = []
chamf = []
print('output path: ', cfg['training']['out_dir'])
for epoch in range(1):
    #TRAIN MODE

    for i, batch in enumerate(val_loader, 0):
        optimizer.zero_grad()
        id, input, gt = batch

        logits, loss = trainer.val_step(batch)
        logger.add_scalar('train/loss', loss, it)
        # val_avg.append(loss.item())
        # np.savez(os.path.join(out_dir, "epoch_40" , str(i)+"data.npz"), pred=logits.detach().cpu().numpy(), inp=input, gt=gt)
        # Print output
        # if print_every > 0 and (it % print_every) == 0:
        t = datetime.datetime.now()
        print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                    % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

        # EVAL
        pred = pred_to_labels(logits)
        np_gt = gt.cpu().numpy()
        # print(occ_pred.shape, occ_gt.shape)

        chamfer_dist = chamfer_distance_voxel_pcd(pred, np_gt)
        class_wise = []
        # print(np.unique(np_gt))
        # print(np.unique(pred))
        for sem in np.unique(pred):
            pred_mask = np.zeros((pred.shape))
            pred_mask[pred == sem] = 1.0
            pred_mask[pred != sem] = 0.0
            gt_mask = np.zeros((gt.shape))
            gt_mask[np_gt == sem] = 1.0
            gt_mask[np_gt != sem] = 0.0
            iou_b = compute_iou(gt_mask, pred_mask)
            if not sum(iou_b) == 0:
                iou_b_clean = [x for x in iou_b if str(x) != 'nan']
                class_wise.append((sum(iou_b_clean) / len(iou_b_clean)))
        val_avg.append((sum(class_wise) / len(class_wise)))

        chamf.append(chamfer_dist)
print(sum(val_avg) / len(val_avg))
print(sum(chamf) / len(chamf))