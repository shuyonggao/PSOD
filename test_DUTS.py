#!/usr/bin/python3
#coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import dataset_1st
from torch.utils.data import DataLoader
from models.modeling import VisionTransformer, get_config


class Test(object):
    def __init__(self, Dataset, Network, Path, snapshot):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot=snapshot, mode='test') 
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)


        config = get_config()
        self.net = Network(config, img_size=352, zero_head=False)  
        self.net.cuda()

        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(self.cfg.snapshot,map_location=torch.device('cpu'))
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if (k.replace('module.', '') in model_dict)}
        
        # check unloaded weights
        for k,v in model_dict.items():
            if k in pretrained_dict.keys():
                pass
            else:
                print("miss keys in pretrained_dict: {}".format(k))

        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

        self.net.train(False)

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape = image.cuda().float(), (H, W)
                image = F.interpolate(image, (352, 352), mode='bilinear', align_corners=True)
                out, _ = self.net(image)
                pred = torch.sigmoid(out[0, 0]).cpu().numpy() * 255
                pred = cv2.resize(pred, dsize=(W,H), interpolation=cv2.INTER_LINEAR)
                # head = './DataStorage/DUTS-TR_transformer/' + self.cfg.datapath.split('/')[-1]   
                head = './DataStorage/DUTS-TR_transformer/'   
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))


if __name__=='__main__':
    for path in ['/home/gaosy/DATA/DUTS/DUTS-TR']:  # 
        # t = Test(dataset_1st, VisionTransformer, path, './out_1st/'+'model-x')
        t = Test(dataset_1st, VisionTransformer, path, './out_1st/'+'model-20')
        t.save()

