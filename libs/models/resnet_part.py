from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer


__all__ = ['ResNetPart', 'resnet18_part', 'resnet34_part', 'resnet50_part', 'resnet101_part',
           'resnet152_part']


class ResNetPart(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 pooling_type='avg', part_pooling_type='avg', num_parts=1):
        assert num_classes == 0, 'Disenable parametric classifier!'
        super(ResNetPart, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.num_parts = num_parts
        # Construct base (pretrained) resnet
        if depth not in ResNetPart.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetPart.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        # self.gap = nn.AdaptiveAvgPool2d(1) # Vanilla model
        self.global_pool = build_pooling_layer(pooling_type) # follow cluster-contrast
        self.part_pool = build_pooling_layer(part_pooling_type) # 局部特征可用其他池化

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                # Global feature
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                
                # Part feature
                self.part_feats = nn.ModuleList([nn.Linear(out_planes, self.num_features) for _ in range(self.num_parts)])
                self.part_bns = nn.ModuleList([nn.BatchNorm1d(self.num_features) for _ in range(self.num_parts)])
                map(lambda part_feat: init.kaiming_normal_(part_feat.weight, mode='fan_out'), self.part_feats)
                map(lambda part_feat: init.constant_(part_feat.bias, 0), self.part_feats)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                
                # 局部特征每个branch的BN
                self.part_bns = nn.ModuleList([nn.BatchNorm1d(self.num_features) for _ in range(self.num_parts)])
            
            # 禁用BN的bias更新
            self.feat_bn.bias.requires_grad_(False) # NOTE BN的bias不被优化，但是weight会被优化
            map(lambda part_bn: part_bn.bias.requires_grad_(False), self.part_bns) # 局部branch的BN也禁用bias的优化
            
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        
        # 初始化BN参数
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)
        map(lambda part_bn: init.constant_(part_bn.weight, 1), self.part_bns)
        map(lambda part_bn: init.constant_(part_bn.bias, 0), self.part_bns)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        bs = x.size(0)
        x = self.base(x) # (b, c, h, w)

        # Part level features
        if self.num_parts > 1:
            part_x = x.split(x.size(2)//self.num_parts, dim=2)[:self.num_parts]
        else:
            part_x = None

        assert part_x is not None, 'Check num_parts!'

        x = self.global_pool(x)
        part_x = list(map(self.part_pool, part_x))
        x = x.view(x.size(0), -1)
        part_x = list(map(lambda part: part.view(part.size(0), -1), part_x))

        if self.cut_at_pooling:
            return {'global': x, 'part': part_x}

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
            # bn_part_x = list(map(self.feat, part_x))
            # bn_part_x = list(map(self.feat_bn, part_x))
            bn_part_x = [part_head(px) for px, part_head in zip(part_x, self.part_feats)]
            bn_part_x = [part_bn(px) for px, part_bn in zip(bn_part_x, self.part_bns)]
        else:
            bn_x = self.feat_bn(x)
            # bn_part_x = list(map(self.feat_bn, part_x)) # BUG 局部特征和全局特征用了同个BN层，存在问题
            bn_part_x = [self.part_bns[i](part_x[i]) for i in range(self.num_parts)]

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            bn_part_x = list(map(F.normalize, bn_part_x))
            return {'global': bn_x, 'part': bn_part_x}

        if self.norm:
            bn_x = F.normalize(bn_x)
            bn_part_x = list(map(F.normalize, bn_part_x))
        elif self.has_embedding:
            bn_x = F.relu(bn_x)
            bn_part_x = list(map(F.relu, bn_part_x))

        if self.dropout > 0:
            bn_x = self.drop(bn_x)
            bn_part_x = list(map(self.drop, bn_part_x))

        # if self.num_classes > 0:
        #     prob = self.classifier(bn_x)
        # else:
        return {'global': bn_x, 'part': bn_part_x}

        # return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNetPart.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())

def resnet18_part(**kwargs):
    return ResNetPart(18, **kwargs)


def resnet34_part(**kwargs):
    return ResNetPart(34, **kwargs)


def resnet50_part(**kwargs):
    return ResNetPart(50, **kwargs)

def resnet101_part(**kwargs):
    return ResNetPart(101, **kwargs)


def resnet152_part(**kwargs):
    return ResNetPart(152, **kwargs)
