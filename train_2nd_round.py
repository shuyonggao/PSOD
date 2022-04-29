#!/usr/bin/python3
#coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

import sys
import datetime

import dataset_2nd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, get_config
from lscloss import *

import  torchvision.utils as vutils
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import random



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def setup(args):
    config = args
    model = VisionTransformer(config, args.img_size, zero_head=False) 
    model.load_from(np.load(args.pretrained_dir))
    num_params = count_parameters(model)
    print(num_params)
    return model

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
def train(Dataset, Network):

    cfg    = Dataset.Config(datapath='/home/gaosy/DATA/Gao_DUTS_TR', savepath='./out_2nd', mode='train', batch=21, lr=0.0025, momen=0.9, decay=5e-4, epoch=20) # batch=28 # lr = 0.03 -》 0.005  0.003
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)  

    ## network
    config =  get_config()
    net = setup(config)

    net = nn.DataParallel(net)
    net.train(True)
    net.cuda()
    
    base, head = [], []
    for name, param in net.named_parameters():
        if 'encoder' in name or 'embeddings' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    CE = torch.nn.BCELoss().cuda()
    loss_lsc = LocalSaliencyCoherence().cuda()

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, gt ,mask, edge, _) in enumerate(loader):
            image, gt, mask, edge = image.type(torch.FloatTensor).cuda(),\
                                           gt.type(torch.FloatTensor).cuda(),\
                                           mask.type(torch.FloatTensor).cuda(), \
                                           edge.type(torch.FloatTensor).cuda()
                                     
            out_final, out_edg = net(image)

            image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_}

            out_final_prob = torch.sigmoid(out_final)
            img_size = image.size(2) * image.size(3) * image.size(0)

            ratio = img_size/ torch.sum(mask)
            sal_loss2 = ratio * CE(out_final_prob*mask, gt*mask)
            out_final_prob = F.interpolate(out_final_prob, scale_factor=0.25, mode='bilinear', align_corners=True)
            # after sigmoid
            loss2_lsc = \
            loss_lsc(out_final_prob, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            edge_loss = 1.0 * CE(torch.sigmoid(out_edg), edge)  

            loss = edge_loss + sal_loss2 + loss2_lsc 
  
            optimizer.zero_grad()
            loss.backward() 
            clip_gradient(optimizer, cfg.lr)
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss_edg':edge_loss.item(), 'loss_final':sal_loss2.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | base_lr=%.6f | loss_edg=%.6f | loss_final=%.6f | loss2_lsc=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], 
                      edge_loss.item(), sal_loss2.item(), loss2_lsc.item()))

            ## tem_see
            tmp_path = './tem_see'
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            if step % 20 == 0:  
                vutils.save_image(torch.sigmoid(out_final[0,:,:,:].data), tmp_path + '/iter%d-sal-final.jpg' % step, normalize=True, padding=0)
                vutils.save_image(image[0,:,:,:].data, tmp_path + '/iter%d-sal-image.jpg' % step, padding=0)  # image[0,:,:,:].data[(2,1,0),:,:]*torch.from_numpy(cfg.std)+torch.from_numpy(cfg.mean)
                vutils.save_image(mask[0,:,:,:].data, tmp_path + '/iter%d-sal-mask.jpg' % step, padding=0)
                vutils.save_image(gt[0,:,:,:].data, tmp_path + '/iter%d-sal-gt.jpg' % step, padding=0)
                vutils.save_image(edge[0,:,:,:].data, tmp_path + '/iter%d-sal-edge.jpg' % step, padding=0)
                vutils.save_image(out_edg[0,:,:,:].data, tmp_path + '/iter%d-sal-out_edge.jpg' % step, padding=0)

        if epoch > 10:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    set_seed(7)
    train(dataset_2nd, VisionTransformer)

