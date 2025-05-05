#!/bin/usr/python3

import os
import shutil
import tempfile
from glob import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torchio as tio
from torchio import SubjectsDataset, SubjectsLoader
from monai.data import DataLoader
import monai
from monai.config import print_config
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from utils.helpers import makedir
from resnet2 import resnet50
from utils.log import create_logger
from torchsummary import summary
import time
import nibabel as nib
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

## Set the seed for reproducibility
set_determinism(seed=0)

## Create a parser
parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', default='/bids_dir')
parser.add_argument('prep_dir', default='/bids_dir/preprocessed')
parser.add_argument('output_dir', default='/out_dir')
parser.add_argument('pretrain_path', default='./pretrain/resnet_50.pth')
parser.add_argument('--n_classes', default='2')
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')  # >>> ADDED <<<

args = parser.parse_args()
bids_dir = args.bids_dir
prep_dir = args.prep_dir
output_dir = args.output_dir
pretrain_path = args.pretrain_path
n_classes = int(args.n_classes)

makedir(output_dir)
log, logclose = create_logger(log_filename=os.path.join(output_dir, 'training.log'))

df_participants = pd.read_csv(os.path.join(bids_dir, "participants.tsv"), sep="\t", dtype=str)
common_subjects_to_analyze = df_participants.participant_id.tolist()
datasets = df_participants.dataset.tolist()
labels = df_participants.label.astype(float).tolist()

def nib_reader(path):
    load_img = nib.load(path)
    img = np.squeeze(load_img.get_fdata())
    img = np.expand_dims(img, axis=0)
    affine = load_img.affine
    load_img.uncache()
    return [torch.tensor(img, dtype=torch.float32), affine]

train_subjects, validation_subjects = [], []
for i, subid in enumerate(common_subjects_to_analyze):
    if "train" in datasets[i]:
        filename = os.path.join(prep_dir, "train", subid + "_prep.nii.gz")
        train_subjects.append(tio.Subject(image=tio.ScalarImage(filename, reader=nib_reader),
                                          label=torch.tensor(labels[i], dtype=torch.float32)))
    elif "val" in datasets[i]:
        validation_subjects.append(tio.Subject(image=tio.ScalarImage(os.path.join(prep_dir, "val", subid + "_prep.nii.gz"),
                                                                     reader=nib_reader),
                                               label=torch.tensor(labels[i], dtype=torch.float32)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = SubjectsDataset(train_subjects)
train_loader = SubjectsLoader(train_dataset, batch_size=2, shuffle=True, num_workers=12, pin_memory=True)
val_dataset = SubjectsDataset(validation_subjects)
val_loader = SubjectsLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)

# --- MODEL SETUP ---
model = resnet50(sample_input_D=256, sample_input_H=256, sample_input_W=256, num_seg_classes=n_classes)
model = nn.Sequential(model, nn.AvgPool3d(32), nn.Flatten(), nn.Linear(2048, 2))  # classifier head
model.to(device)

loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

start_epoch = 0  # >>> ADDED <<<

if args.resume:  # >>> ADDED <<<
    log(f"Resuming training from checkpoint: {pretrain_path}")
    checkpoint = torch.load(pretrain_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    log('Loading pretrained model weights selectively (backbone)')
    pretrained_weights = torch.load(pretrain_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# Selective layer freezing (if needed)
for name, param in model.named_parameters():
    if "0.layer4" in name:  # since it's now in Sequential, prefix is "0."
        param.requires_grad = True
    else:
        param.requires_grad = False

val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

# --- TRAINING LOOP ---
for epoch in range(start_epoch, 43):
    log("-" * 10)
    log(f"epoch {epoch + 1}")
    model.train()
    epoch_loss = 0
    step = 0
    start_time = time.time()

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"]["data"].to(device), batch_data["label"].long().to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_loader.batch_size
        log(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    elapsed_time = time.time() - start_time
    log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    log("Epoch time duration: " + str(elapsed_time))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data["image"]['data'].to(device), val_data["label"].long().to(device)
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
            log(f"current epoch: {epoch + 1} current accuracy: {metric:.4f} best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

    if (epoch + 1) % 2 == 0:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)  # >>> CHANGED <<<

log(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
logclose()
