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
# train_dataset = config.get_dataset('train', cfg)
# val_dataset = config.get_dataset('val', cfg, return_idx=True)

train_dataset = Kitti360(cfg, train=True, npoints=cfg['data']['pointcloud_n'])
val_dataset = Kitti360(cfg, train=False, npoints=cfg['data']['pointcloud_n'])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, num_workers=cfg['training']['n_workers_val'], shuffle=False)

# # For visualizations
# vis_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=1, shuffle=False,)
# model_counter = defaultdict(int)
# data_vis_list = []

# # Build a data dictionary for visualization
# iterator = iter(vis_loader)
# for i in range(len(vis_loader)):
#     data_vis = next(iterator)
#     idx = data_vis['idx'].item()
#     model_dict = val_dataset.get_model_dict(idx)
#     category_id = model_dict.get('category', 'n/a')
#     category_name = val_dataset.metadata[category_id].get('name', 'n/a')
#     category_name = category_name.split(',')[0]
#     if category_name == 'n/a':
#         category_name = category_id

#     c_it = model_counter[category_id]
#     if c_it < vis_n_outputs:
#         data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

#     model_counter[category_id] += 1

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
    load_dict = checkpoint_io.load('149model.pt')
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

print('output path: ', cfg['training']['out_dir'])
for epoch in range(1):
    #TRAIN MODE

    for i, batch in enumerate(train_loader, 0):
        optimizer.zero_grad()
        id, input, gt = batch

        logits, loss = trainer.val_step(batch)
        logger.add_scalar('train/loss', loss, it)
        
        np.savez(os.path.join(out_dir, "train_data" , str(i)+"data.npz"), pred=logits.numpy(), inp=input, gt=gt)
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))