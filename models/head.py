import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mmcv.cnn import build_norm_layer
import math


class VisionTransformerUpHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, embed_dim=1024,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg= {'type': 'BN', 'requires_grad': True},
                 num_conv=4, upsampling_method='bilinear', num_upsampe_layer=4):
        super(VisionTransformerUpHead, self).__init__()
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.num_conv = num_conv
        self.norm = norm_layer(embed_dim)
        self.upsampling_method = upsampling_method
        self.num_upsampe_layer = num_upsampe_layer

        self.conv_0 = nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # 256


        self.edge_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.edge_out = nn.Conv2d(256, 1, 3, 1, 1)  

        self.conv_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.conv_4 = nn.Sequential(nn.Conv2d(256, 64, 5, 1, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 1, 3, 1, padding=1))

        _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc_2 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_edge = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc_3 = build_norm_layer(self.norm_cfg, 256)

        self.fuse_side = nn.Conv2d(512, 256, kernel_size=1, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, res_feat):
        # x = self._transform_inputs(x) 

        x = x[-1][: ,1:]
        x = self.norm(x)
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)

        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x,inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = self.conv_1(x)
        x = self.syncbn_fc_1(x)
        x = F.relu(x,inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        res_feat = F.interpolate(res_feat[0], size=(4*h, 4*w), mode='bilinear', align_corners=False)
        edge_feat = torch.cat((x,res_feat), dim=1)
        edge_feat = self.edge_conv(edge_feat)
        edge_feat = self.syncbn_edge(edge_feat)
        edge_feat = F.relu(edge_feat,inplace=True)
        edge_out = self.edge_out(edge_feat)
        edge_out = F.interpolate(edge_out, size=x.shape[-1]*4, mode='bilinear', align_corners=False)
        x = self.conv_2(x)
        x = self.syncbn_fc_2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, edge_feat), dim=1)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = self.conv_3(x)
        x = self.syncbn_fc_3(x)
        x = F.relu(x,inplace=True)
        x = self.conv_4(x)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=False)

        return x, edge_out
