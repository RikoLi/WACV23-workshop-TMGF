"""
The backbone implementation is inspired by TransReID series. The pretrained weighted is provided by LUPerson.
Thanks for their excellent works!
TransReID: https://github.com/damo-cv/TransReID
TransReID-SSL: https://github.com/damo-cv/TransReID-SSL
LUPerson: https://github.com/DengpanFu/LUPerson
"""

import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from libs.models.vit import vit_small_patch16_224_TransReID

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

        
class TMGF(nn.Module):
    """
    Transformer-based Multi-Grained Feature encoder.
    """
    
    __factory = {
        'tmgf': vit_small_patch16_224_TransReID
    }
    
    def __init__(self, arch, img_size, sie_coef, camera_num, view_num, stride_size, drop_path_rate, drop_rate, attn_drop_rate,
                 pretrain_path, hw_ratio, gem_pool, stem_conv, num_parts, has_early_feature, has_head, global_feature_type,
                 granularities, branch, enable_early_norm, **kwargs):
        super().__init__()
        print(f'using Transformer_type: {arch} as a backbone')
        
        assert sum(granularities) == num_parts
        assert branch in ('all', 'b1', 'b2')

        if camera_num:
            camera_num = camera_num
        else:
            camera_num = 0
        if view_num:
            view_num = view_num
        else:
            view_num = 0

        self.base = TMGF.__factory[arch](img_size=img_size, sie_xishu=sie_coef, camera=camera_num, view=view_num, stride_size=stride_size,
                                           drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                           gem_pool=gem_pool, stem_conv=stem_conv, has_early_feature=has_early_feature,
                                           enable_early_norm=enable_early_norm, **kwargs) # local_feature = True for no projection head ablation
        self.in_planes = self.base.in_planes
        self.has_head = has_head
        self.global_feature_type = global_feature_type
        self.granularities = granularities
        self.branch = branch
        
        
        if pretrain_path != '':
            if osp.exists(pretrain_path):
                self.base.load_param(pretrain_path, hw_ratio)
                print('Loading pretrained weights from {} ...'.format(pretrain_path))
            else:
                raise FileNotFoundError('Cannot find {}'.format(pretrain_path))
        else:
            print('Initialize weights randomly.')
            
        # Part split settings
        self.num_parts = num_parts
        self.fmap_h = img_size[0] // stride_size[0]
        self.fmap_w = img_size[1] // stride_size[1]
        
        # Two different granularity branches
        if self.has_head:
            block = self.base.blocks[-1]
            layer_norm = self.base.norm
            self.b1 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            self.b2 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
        
        # Pooling layers
        for i, g in enumerate(self.granularities):
            setattr(self, 'b{}_pool'.format(i+1), nn.AvgPool2d(kernel_size=(self.fmap_h//g, self.fmap_w),
                                                               stride=(self.fmap_h//g,)))
            
        print('num_parts={}, branch_parts={}'.format(self.num_parts, self.granularities))

        # Global bottleneck
        self.bottleneck = self.make_bnneck(self.in_planes, weights_init_kaiming)
        
        
        # Part bottleneck
        self.part_bns = nn.ModuleList([
            self.make_bnneck(self.in_planes, weights_init_kaiming) for i in range(self.num_parts)
        ])
        
    def forward_single_branch(self, x, branch, label=None, cam_label=None, view_label=None):
        """
        Full ViT, no projection head. One part pooling branch.
        """
        
        x = self.base(x, cam_label=cam_label, view_label=view_label)
        B = x.size(0)
        x_glb = x[:,0,:]
        x_patch = x[:,1:,:]
        x_patch = x_patch.permute(0,2,1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_part = getattr(self, '{}_pool'.format(branch))(x_patch).squeeze()
        
        return x_glb, x_part
    
    def forward_multi_branch(self, x, label=None, cam_label=None, view_label=None):
        """
        ViT 1st ~ (L-1)-th layers, duplicated L-th layers as projection heads for two branches.
        """
        
        x = self.base(x, cam_label=cam_label, view_label=view_label) # output before last layer
        B = x.size(0)
        
        # Split after head
        # branch 1
        x_b1 = self.b1(x) # (B, L, C)
        x_b1_glb = x_b1[:,0,:] # (B, C)
        x_b1_patch = x_b1[:,1:,:] # (B, L-1, C)
        x_b1_patch = x_b1_patch.permute(0,2,1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_b1_patch = self.b1_pool(x_b1_patch).squeeze() # (B, C, P1)
        
        # branch 2
        x_b2 = self.b2(x)
        x_b2_glb = x_b2[:,0,:]
        x_b2_patch = x_b2[:,1:,:]
        x_b2_patch = x_b2_patch.permute(0,2,1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_b2_patch = self.b2_pool(x_b2_patch).squeeze() # (B, C, P2)
    
        # Mean global feature
        if self.global_feature_type == 'mean':
            x_glb = 0.5 * (x_b1_glb + x_b2_glb) # (B, C)
        elif self.global_feature_type == 'b1':
            x_glb = x_b1_glb
        elif self.global_feature_type == 'b2':
            x_glb = x_b2_glb
        else:
            raise ValueError('Invalid global feature type: {}'.format(self.global_feature_type))
        
        # Stack two branch part features
        x_part = torch.cat([x_b1_patch, x_b2_patch], dim=2) # (B, C, P), P = P1 + P2
        
        return x_glb, x_part
    
    def forward_multi_branch_no_head(self, x, label=None, cam_label=None, view_label=None):
        """
        Full ViT, no projection head. Two part pooling branches.
        """
        
        x = self.base(x, cam_label=cam_label, view_label=view_label)
        B = x.size(0)
        
        # Split without head
        # branch 1
        x_patch = x[:,1:,:] # (B, L-1, C)
        x_patch = x_patch.permute(0,2,1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_b1_patch = self.b1_pool(x_patch).squeeze() # (B, C, P1)
        
        # branch 2
        x_b2_patch = self.b2_pool(x_patch).squeeze() # (B, C, P2)
        
        # global feature
        x_glb = x[:,0,:] # (B, C)
        
        # Stack two branch part features
        x_part = torch.cat([x_b1_patch, x_b2_patch], dim=2) # (B, C, P), P = P1 + P2
        
        return x_glb, x_part
            
    def forward(self, x, label=None, cam_label=None, view_label=None):
        B = x.size(0)
        if self.has_head:
            x_glb, x_part = self.forward_multi_branch(x, label, cam_label, view_label)
        elif self.branch != 'all':
            x_glb, x_part = self.forward_single_branch(x, self.branch, label, cam_label, view_label)
        else:
            x_glb, x_part = self.forward_multi_branch_no_head(x, label, cam_label, view_label)
        
        # BNNeck + L2 norm
        x_glb = self.bottleneck(x_glb)
        x_part = torch.stack([self.part_bns[i](x_part[:,:,i]) for i in range(x_part.size(2))], dim=2)
        
        x_glb = F.normalize(x_glb, dim=1)
        x_part = F.normalize(x_part, dim=1)
        
        assert x_part.size(2) == self.num_parts, 'x_part size: {} != num_parts: {}'.format(
            x_part.size(2), self.num_parts) # check part num
        
        return {'global': x_glb, 'part': x_part.permute(2,0,1)} # x_part as (P, B, C)
    
    def make_bnneck(self, dims, init_func):
        bn = nn.BatchNorm1d(dims)
        bn.bias.requires_grad_(False) # disable bias update
        bn.apply(init_func)
        return bn

def tmgf(**kwargs):
    return TMGF(**kwargs)