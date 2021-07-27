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

args = parser.parse_args()
cfg = config.load_config(args.config)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
num_workers = cfg['training']['n_workers']
epochs = cfg['training']['epochs']
# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_dataset = Kitti360(dataset_path=cfg['data']['path'], split=cfg['data']['train_split'], pose_path=cfg['data']['pose_path'], train=True, weights=False , npoints_partial = 4096, npoints=8192)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers, drop_last=True)
val_dataset = Kitti360(dataset_path=cfg['data']['path'], split=cfg['data']['val_split'], pose_path=cfg['data']['pose_path'], train=False, weights=False, npoints_partial = 4096, npoints=8192)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size,
                                        shuffle=False, num_workers=num_workers, drop_last=True)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt') # LOAD MODEL HERE

except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)

logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])
for epoch in range(0,epochs):
    #TRAIN MODE
    for i, batch in enumerate(train_loader, 0):
        optimizer.zero_grad()
        id, input, gt = batch

        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)

        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch, it, loss, time.time() - t0, t.hour, t.minute))

        it+=1

    if epoch % 10 == 0:
        checkpoint_io.save(str(epoch)+'model.pt', epoch_it=epoch, it=it)
