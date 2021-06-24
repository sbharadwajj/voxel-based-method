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

def pred_to_labels(logits):
    sm = torch.nn.Softmax(dim=1)
    softmax_pred = sm(logits).cpu().numpy()
    pred = np.argmax(softmax_pred, axis=1)
    return pred

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

    # Convert to boolean values
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
        occ_pred = pred_to_labels(logits)
        occ_gt = gt.cpu().numpy()
        # print(occ_pred.shape, occ_gt.shape)
        iou_b = compute_iou(occ_gt, occ_pred)
        val_avg.append(iou_b)
print(sum(val_avg) / len(val_avg))