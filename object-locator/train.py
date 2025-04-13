from __future__ import print_function

__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"


import math
import cv2
import os
import sys
import time
import shutil
from itertools import chain
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import skimage.transform
from peterpy import peter
from ballpark import ballpark

from . import losses
from .models import unet_model
from .metrics import Judge
from . import logger
from . import argparser
from . import utils
from . import data
from .data import csv_collator
from .data import RandomHorizontalFlipImageAndLabel
from .data import RandomVerticalFlipImageAndLabel
from .data import ScaleImageAndLabel
from .comparison import analyze_alpha_performance
import torch.nn.functional as F

def find_blob_centroids_gpu(mask, min_area=5):
    """
    Parameters:
        mask: torch.Tensor (H, W), value range [0, 1]
    Returns:
        centroids: torch.Tensor (N, 2), coordinate format [y, x]
    """
    mask = (mask > 0.5).float()
    H, W = mask.shape
    
    # Generate coordinate grid for each pixel
    y = torch.arange(H, device=mask.device)
    x = torch.arange(W, device=mask.device)
    y_coords, x_coords = torch.meshgrid(y, x)
    
    # Label connected regions (simplified version, might not be as accurate as OpenCV)
    labeled = torch.zeros_like(mask, dtype=torch.long)
    current_label = 1
    # Use PyTorch operations to replace loops
    while True:
        unlabeled = (mask > 0.5) & (labeled == 0)
        if not unlabeled.any():
            break
        labeled[unlabeled] = current_label
        current_label += 1
    
    # Compute the centroid of each connected region
    centroids = []
    for label in range(1, current_label):
        region_mask = (labeled == label)
        if region_mask.sum() >= min_area:
            y_center = (y_coords * region_mask).sum() / region_mask.sum()
            x_center = (x_coords * region_mask).sum() / region_mask.sum()
            centroids.append(torch.stack([y_center, x_center]))
    
    if not centroids:
        return torch.zeros((0, 2), dtype=torch.float32, device=mask.device)
    return torch.stack(centroids)

def gpu_paint_circles(img_tensor, points, color='white', radius=3):
    """
    img_tensor: (C, H, W) image tensor on GPU
    points: (N, 2) point coordinates [y, x] on GPU
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
    # Create a zero tensor of the same size as the image
    circle_layer = torch.zeros_like(img_tensor[:, :1])
    
    # Draw circles for each point
    for y, x in points:
        y = torch.arange(img_tensor.size(2), device=img_tensor.device)
        x = torch.arange(img_tensor.size(3), device=img_tensor.device)
        yy, xx = torch.meshgrid(y, x)
        dist = ((yy - y)**2 + (xx - x)**2).sqrt()
        circle = (dist <= radius).float()
        circle_layer += circle.unsqueeze(0).unsqueeze(0)
    
    # Set channel values based on color
    if color == 'white':
        color_tensor = torch.tensor([1., 1., 1.], device=img_tensor.device)
    elif color == 'red':
        color_tensor = torch.tensor([1., 0., 0.], device=img_tensor.device)
    
    # Overlay the circles onto the original image
    color_tensor = color_tensor.view(3, 1, 1)  # Broadcasting mechanism
    img_tensor = torch.where(circle_layer.unsqueeze(1) > 0, color_tensor, img_tensor)
    
    return img_tensor.squeeze(0)


def train_one_epoch(model, optimizer, loss_fn, train_loader, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if args.fp16:
            imgs = imgs.half()
            targets = targets.half()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    return running_loss / len(train_loader)

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs):
        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# Parse command line arguments
args = argparser.parse_command_args('training')

# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')
device = torch.device('cuda') if args.cuda else device_cpu

# Create directory for checkpoint to be saved
if args.save:
    os.makedirs(os.path.split(args.save)[0], exist_ok=True)

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Visdom setup
log = logger.Logger(server=args.visdom_server,
                    port=args.visdom_port,
                    env_name=args.visdom_env)


# Create data loaders (return data in batches)
trainset_loader, valset_loader = \
    data.get_train_val_loaders(train_dir=args.train_dir,
                               max_trainset_size=args.max_trainset_size,
                               collate_fn=csv_collator,
                               height=args.height,
                               width=args.width,
                               seed=args.seed,
                               batch_size=args.batch_size * 2,  # 验证时可增大batch_size
                               drop_last_batch=args.drop_last_batch,
                               num_workers=min(4, os.cpu_count()),  # 根据CPU核心数调整
                               val_dir=args.val_dir,
                               pin_memory=args.cuda and torch.cuda.is_available(),                   # 加速CPU->GPU传输
                               # persistent_workers=True            # 保持worker进程（PyTorch 1.7+）
                               max_valset_size=args.max_valset_size)

# Model
with peter('Building network'):
    model = unet_model.UNet(3, 1,
                            height=args.height,
                            width=args.width,
                            known_n_points=args.n_points,
                            device=device,
                            ultrasmall=args.ultrasmallnet)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" with {ballpark(num_params)} trainable parameters. ", end='')
model = nn.DataParallel(model)
model.to(device)

# Loss functions
loss_regress = nn.SmoothL1Loss()
loss_loc = losses.WeightedHausdorffDistance(resized_height=args.height,
                                            resized_width=args.width,
                                            p=args.p,
                                            return_2_terms=True,
                                            device=device)

# Optimization strategy
if args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           amsgrad=True)

start_epoch = 0
lowest_mahd = np.infty

# Restore saved checkpoint (model weights + epoch + optimizer state)
if args.resume:
    with peter('Loading checkpoint'):
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            try:
                lowest_mahd = checkpoint['mahd']
            except KeyError:
                lowest_mahd = np.infty
                print('W: Loaded checkpoint has not been validated. ', end='')
            model.load_state_dict(checkpoint['model'])
            if not args.replace_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"\n\__ loaded checkpoint '{args.resume}'"
                  f"(now on epoch {checkpoint['epoch']})")
        else:
            print(f"\n\__ E: no checkpoint found at '{args.resume}'")
            exit(-1)

running_avg = utils.RunningAverage(len(trainset_loader))

normalzr = utils.Normalizer(args.height, args.width)

# Time at the last evaluation
tic_train = -np.infty
tic_val = -np.infty

epoch = start_epoch
it_num = 0
while epoch < args.epochs:

    loss_avg_this_epoch = 0
    iter_train = tqdm(trainset_loader,
                      desc=f'Epoch {epoch} ({len(trainset_loader.dataset)} images)')

    # === TRAIN ===

    # Set the module in training mode
    model.train()

    for batch_idx, (imgs, dictionaries) in enumerate(iter_train):

        # Pull info from this batch and move to device
        imgs = imgs.to(device)
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_counts = [dictt['count'].to(device)
                         for dictt in dictionaries]
        target_orig_heights = [dictt['orig_height'].to(device)
                               for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'].to(device)
                              for dictt in dictionaries]

        # Lists -> Tensor batches
        target_counts = torch.stack(target_counts)
        target_orig_heights = torch.stack(target_orig_heights)
        target_orig_widths = torch.stack(target_orig_widths)
        target_orig_sizes = torch.stack((target_orig_heights,
                                         target_orig_widths)).transpose(0, 1)

        # One training step
        optimizer.zero_grad()
        est_maps, est_counts = model.forward(imgs)
        term1, term2 = loss_loc.forward(est_maps,
                                        target_locations,
                                        target_orig_sizes)
        est_counts = est_counts.view(-1)
        target_counts = target_counts.view(-1)
        term3 = loss_regress.forward(est_counts, target_counts)
        term3 *= args.lambdaa
        loss = term1 + term2 + term3
        loss.backward()
        optimizer.step()

        # Update progress bar
        running_avg.put(loss.item())
        iter_train.set_postfix(running_avg=f'{round(running_avg.avg/3, 1)}')

        # Log training error
        if time.time() > tic_train + args.log_interval:
            tic_train = time.time()

            # Log training losses
            log.train_losses(terms=[term1, term2, term3, loss / 3, running_avg.avg / 3],
                             iteration_number=epoch +
                             batch_idx/len(trainset_loader),
                             terms_legends=['Term1',
                                            'Term2',
                                            'Term3*%s' % args.lambdaa,
                                            'Sum/3',
                                            'Sum/3 runn avg'])

            # Resize images to original size
            orig_shape = target_orig_sizes[0].data.to(device_cpu).numpy().tolist()
            '''orig_img_origsize = ((skimage.transform.resize(imgs[0].data.squeeze().to(device_cpu).numpy().transpose((1, 2, 0)),
                                                           output_shape=orig_shape,
                                                           mode='constant') + 1) / 2.0 * 255.0).\
                astype(np.float32).transpose((2, 0, 1))'''
    
            
            # 使用PyTorch的插值函数替代skimage.transform.resize
            orig_img_origsize = F.interpolate(
                imgs[0].unsqueeze(0),  # 添加batch维度
                size=tuple(orig_shape),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            orig_img_origsize = ((orig_img_origsize + 1) / 2.0 * 255.0).clamp(0, 255)


            est_map_origsize = skimage.transform.resize(est_maps[0].data.unsqueeze(0).to(device_cpu).numpy().transpose((1, 2, 0)),
                                                        output_shape=orig_shape,
                                                        mode='constant').\
                astype(np.float32).transpose((2, 0, 1)).squeeze(0)

            # Overlay output on heatmap
            orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                map=est_map_origsize).\
                astype(np.float32)

            # Send heatmap with circles at the labeled points to Visdom
            target_locs_np = target_locations[0].\
                to(device_cpu).numpy().reshape(-1, 2)
            target_orig_size_np = target_orig_sizes[0].\
                to(device_cpu).numpy().reshape(2)
            target_locs_wrt_orig = normalzr.unnormalize(target_locs_np,
                                                        orig_img_size=target_orig_size_np)
            img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                               points=target_locs_wrt_orig,
                                               color='white')
            log.image(imgs=[img_with_x],
                      titles=['(Training) Image w/ output heatmap and labeled points'],
                      window_ids=[1])

            # # Read image with GT dots from disk
            # gt_img_numpy = skimage.io.imread(
            #     os.path.join('/home/jprat/projects/phenosorg/data/plant_counts_dots/20160613_F54_training_256x256_white_bigdots',
            #                  dictionary['filename'][0]))
            # # dots_img_tensor = torch.from_numpy(gt_img_numpy).permute(
            # # 2, 0, 1)[0, :, :].type(torch.FloatTensor) / 255
            # # Send GT image to Visdom
            # viz.image(np.moveaxis(gt_img_numpy, 2, 0),
            #           opts=dict(title='(Training) Ground Truth'),
            #           win=3)

        it_num += 1
        # Add alpha analysis after validation is complete
        if isinstance(args.alpha_values, (list, tuple)) and len(args.alpha_values) > 1:
            if (epoch + 1) % args.alpha_analysis_freq == 0:
                # Save the current alpha value
                original_alpha = model.module.loss_loc.p.item()
                
                alpha_results = []
                for alpha in args.alpha_values:
                    # Set the new alpha value
                    model.module.loss_loc.p.data.fill_(alpha)
                    
                    # Evaluate on the validation set
                    alpha_judge = Judge(r=args.radius)
                    with torch.no_grad():
                        for imgs, dictionaries in valset_loader:
                            # Prepare data...
                            imgs = imgs.to(device)
                            target_locations = [d['locations'].to(device) for d in dictionaries]
                            target_orig_sizes = torch.stack([
                                torch.stack([d['orig_height'].to(device), 
                                        d['orig_width'].to(device)])
                                for d in dictionaries
                            ])
                            
                            # Forward pass
                            est_maps, _ = model(imgs)
                            centroids = find_blob_centroids_gpu(est_maps[0])
                            
                            # Compute metrics
                            for dict_idx in range(len(dictionaries)):
                                gt_points = normalzr.unnormalize(
                                    target_locations[dict_idx].cpu().numpy(),
                                    orig_img_size=target_orig_sizes[dict_idx].cpu().numpy()
                                )
                                alpha_judge.feed_points(centroids.cpu().numpy(), gt_points,
                                                    max_ahd=loss_loc.max_dist)
                    
                    # Record results
                    alpha_results.append({
                        'epoch': epoch,
                        'alpha': alpha,
                        'precision': alpha_judge.precision,
                        'recall': alpha_judge.recall,
                        'fscore': alpha_judge.fscore,
                        'mahd': alpha_judge.mahd
                    })
                
                # Restore the original alpha value
                model.module.loss_loc.p.data.fill_(original_alpha)
                
                # Save results
                alpha_df = pd.DataFrame(alpha_results)
                alpha_file = os.path.join(os.path.split(args.save)[0], f'alpha_results_epoch{epoch}.csv')
                alpha_df.to_csv(alpha_file, index=False)
                
                # Print results
                print(f"\nAlpha analysis at epoch {epoch}:")
                print(alpha_df.to_string(index=False))

    # Never do validation?
    if not args.val_dir or \
            not valset_loader or \
            len(valset_loader) == 0 or \
            args.val_freq == 0:

        # Time to save checkpoint?
        if args.save and (epoch + 1) % args.val_freq == 0:
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'n_points': args.n_points,
                        }, args.save)
        epoch += 1
        continue

    # Time to do validation?
    if (epoch + 1) % args.val_freq != 0:
        epoch += 1
        continue

    # === VALIDATION ===

    # Set the module in evaluation mode
    model.eval()

    judge = Judge(r=args.radius)
    sum_term1 = 0
    sum_term2 = 0
    sum_term3 = 0
    sum_loss = 0
    iter_val = tqdm(valset_loader,
                    desc=f'Validating Epoch {epoch} ({len(valset_loader.dataset)} images)')
    for batch_idx, (imgs, dictionaries) in enumerate(iter_val):

        # Pull info from this batch and move to device
        imgs = imgs.to(device)
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_counts = [dictt['count'].to(device)
                        for dictt in dictionaries]
        target_orig_heights = [dictt['orig_height'].to(device)
                               for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'].to(device)
                              for dictt in dictionaries]

        with torch.no_grad():
            target_counts = torch.stack(target_counts)
            target_orig_heights = torch.stack(target_orig_heights)
            target_orig_widths = torch.stack(target_orig_widths)
            target_orig_sizes = torch.stack((target_orig_heights,
                                             target_orig_widths)).transpose(0, 1)
        orig_shape = (dictionaries[0]['orig_height'].item(),
                      dictionaries[0]['orig_width'].item())

        # Tensor -> float & numpy
        target_count_int = int(round(target_counts.item()))
        target_locations_np = \
            target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
        target_orig_size_np = \
            target_orig_sizes[0].to(device_cpu).numpy().reshape(2)

        normalzr = utils.Normalizer(args.height, args.width)

        if target_count_int == 0:
            continue

        # Feed-forward
        with torch.no_grad():
            est_maps, est_counts = model.forward(imgs)

        # Tensor -> int
        est_count_int = int(round(est_counts.item()))

        # The 3 terms
        with torch.no_grad():
            est_counts = est_counts.view(-1)
            target_counts = target_counts.view(-1)
            term1, term2 = loss_loc.forward(est_maps,
                                            target_locations,
                                            target_orig_sizes)
            term3 = loss_regress.forward(est_counts, target_counts)
            term3 *= args.lambdaa
        sum_term1 += term1.item()
        sum_term2 += term2.item()
        sum_term3 += term3.item()
        sum_loss += term1 + term2 + term3

        # Update progress bar
        loss_avg_this_epoch = sum_loss.item() / (batch_idx + 1)
        iter_val.set_postfix(
            avg_val_loss_this_epoch=f'{loss_avg_this_epoch:.1f}-----')

        # The estimated map must be thresholed to obtain estimated points
        # BMM thresholding
        '''est_map_numpy = est_maps[0, :, :].to(device_cpu).numpy()
        est_map_numpy_origsize = skimage.transform.resize(est_map_numpy,
                                                          output_shape=orig_shape,
                                                          mode='constant')
        mask, _ = utils.threshold(est_map_numpy_origsize, tau=-1)'''

        # GPU加速实现（放在validation循环内）
        with torch.no_grad():
            # 1. 热力图缩放
            est_maps_resized = F.interpolate(
                est_maps.unsqueeze(1), 
                size=tuple(target_orig_sizes[0].int().tolist()[::-1]),  # (width,height)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # 2. 阈值处理
            masks = (est_maps_resized > 0.5).float()
            
            # 3. 聚类中心计算
            centroids = find_blob_centroids_gpu(est_maps_resized[0], min_area=5)  # 处理第一个样本

        '''# Obtain centroids of the mask
        centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                           max_mask_pts=args.max_mask_pts)
'''

        centroids_wrt_orig = centroids  # 直接使用find_blob_centroids_gpu的结果
        # assert masks.dim() == 4, f"Expected [B,C,H,W], got {masks.shape}"

        # Validation metrics
        target_locations_wrt_orig = normalzr.unnormalize(target_locations_np,
                                                         orig_img_size=target_orig_size_np)
        judge.feed_points(centroids_wrt_orig, target_locations_wrt_orig,
                          max_ahd=loss_loc.max_dist)
        judge.feed_count(est_count_int, target_count_int)

        if time.time() > tic_val + args.log_interval:
            tic_val = time.time()

            # Resize to original size
            orig_img_origsize = ((skimage.transform.resize(imgs[0].to(device_cpu).squeeze().numpy().transpose((1, 2, 0)),
                                                           output_shape=target_orig_size_np.tolist(),
                                                           mode='constant') + 1) / 2.0 * 255.0).\
                astype(np.float32).transpose((2, 0, 1))
            est_map_origsize = skimage.transform.resize(est_maps[0].to(device_cpu).unsqueeze(0).numpy().transpose((1, 2, 0)),
                                                        output_shape=orig_shape,
                                                        mode='constant').\
                astype(np.float32).transpose((2, 0, 1)).squeeze(0)

            # Overlay output on heatmap
            orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                map=est_map_origsize).\
                astype(np.float32)

            # # Read image with GT dots from disk
            # gt_img_numpy = skimage.io.imread(
            #     os.path.join('/home/jprat/projects/phenosorg/data/plant_counts_dots/20160613_F54_validation_256x256_white_bigdots',
            #                  dictionary['filename'][0]))
            # # dots_img_tensor = torch.from_numpy(gt_img_numpy).permute(
            #     # 2, 0, 1)[0, :, :].type(torch.FloatTensor) / 255
            # # Send GT image to Visdom
            # viz.image(np.moveaxis(gt_img_numpy, 2, 0),
            #           opts=dict(title='(Validation) Ground Truth'),
            #           win=7)
            if not args.paint:
                # Send input and output heatmap (first one in the batch)
                log.image(imgs=[orig_img_w_heatmap_origsize],
                          titles=['(Validation) Image w/ output heatmap'],
                          window_ids=[5])
            else:
                # Send heatmap with a cross at the estimated centroids to Visdom
                '''img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                 points=centroids_wrt_orig,
                                                 color='red',
                                                 crosshair=True )'''
                
                # 在GPU上绘制标记点
                img_with_x = gpu_paint_circles(orig_img_w_heatmap_origsize,
                                            points=target_locs_wrt_orig,
                                            color='white')

                log.image(imgs=[img_with_x],
                          titles=['(Validation) Image w/ output heatmap '
                                  'and point estimations'],
                          window_ids=[8])

    avg_term1_val = sum_term1 / len(valset_loader)
    avg_term2_val = sum_term2 / len(valset_loader)
    avg_term3_val = sum_term3 / len(valset_loader)
    avg_loss_val = sum_loss / len(valset_loader)

    # Log validation metrics
    log.val_losses(terms=(avg_term1_val,
                          avg_term2_val,
                          avg_term3_val,
                          avg_loss_val / 3,
                          judge.mahd,
                          judge.mae,
                          judge.rmse,
                          judge.mape,
                          judge.coeff_of_determination,
                          judge.pearson_corr \
                              if not np.isnan(judge.pearson_corr) else 1,
                          judge.precision,
                          judge.recall),
                   iteration_number=epoch,
                   terms_legends=['Term 1',
                                  'Term 2',
                                  'Term3*%s' % args.lambdaa,
                                  'Sum/3',
                                  'AHD',
                                  'MAE',
                                  'RMSE',
                                  'MAPE (%)',
                                  'R^2',
                                  'r',
                                  f'r{args.radius}-Precision (%)',
                                  f'r{args.radius}-Recall (%)'])

    # If this is the best epoch (in terms of validation error)
    if judge.mahd < lowest_mahd:
        # Keep the best model
        lowest_mahd = judge.mahd
        if args.save:
            torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                        'model': model.state_dict(),
                        'mahd': lowest_mahd,
                        'optimizer': optimizer.state_dict(),
                        'n_points': args.n_points,
                        }, args.save)
            print("Saved best checkpoint so far in %s " % args.save)

    epoch += 1


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
