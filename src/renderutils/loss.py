# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torchmetrics
#----------------------------------------------------------------------------
# HDR image losses
#----------------------------------------------------------------------------

def _tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)

def _SMAPE(img, target, eps=0.01):
    nom = torch.abs(img - target)
    denom = torch.abs(img) + torch.abs(target) + 0.01
    return torch.mean(nom / denom)

def _RELMSE(img, target, eps=0.1):
    nom = (img - target) * (img - target)
    denom = img * img + target * target + 0.1 
    return torch.mean(nom / denom)


def _cie76(img, target, mask):
    # b h w c to b c h w
    img = img.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    print
    sdi = torchmetrics.image.UniversalImageQualityIndex().to('cuda')
    loss = sdi(img, target)
    # print('loss', loss)
    return  loss

def image_loss_fn(img, target, loss, tonemapper, mask=None):
    
    if tonemapper == 'log_srgb':
        img    = _tonemap_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
        target = _tonemap_srgb(torch.log(torch.clamp(target, min=0, max=65535) + 1))

    if mask is not None:
        if 1: # mask all pixels, the shape is M
            img = torch.masked_select(img, mask.unsqueeze(-1))
            target = torch.masked_select(target, mask.unsqueeze(-1))

    if loss == 'mse':
        return torch.nn.functional.mse_loss(img, target)
    elif loss == 'smape':
        return _SMAPE(img, target)
    elif loss == 'relmse':
        return _RELMSE(img, target)
    elif loss == 'smoothl1':
        return torch.nn.functional.smooth_l1_loss(img, target)
    elif loss == 'huber':
        return torch.nn.functional.huber_loss(img, target)
    else:    
        return torch.nn.functional.l1_loss(img, target)
        

# test _cie76

# _cie76(None, None, None)
# print('test _cie76 passed')