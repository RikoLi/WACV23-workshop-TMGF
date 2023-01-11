from __future__ import absolute_import
from collections import defaultdict
import math

from typing import *
import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4, class_position=1):
        self.data_source = data_source
        #self.class_position = class_posotion
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        #for index, (_, pid, _) in enumerate(data_source):
        for index, each_input in enumerate(data_source):
            pid = each_input[class_position]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, class_position, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances
        self.class_position = class_position
        
        #for index, (_, pid, cam) in enumerate(data_source):
        for index, each_input in enumerate(data_source):
            pid = each_input[self.class_position] # 1: cluster_label, 4: proxy_label
            cam = each_input[2]
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            #_, i_pid, i_cam = self.data_source[i]
            i_pid = self.data_source[i][1]
            i_cam = self.data_source[i][2]
            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:  # as a priority: select images in the same cluster/class, from different cameras (my add)

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:  # otherwise select images in the same camera, or do not select more if it's an outlier (my add)
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)



class ClassUniformlySampler(Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''
    def __init__(self, samples, class_position, k=4, has_outlier=False, cam_num=0):

        self.samples = samples
        self.class_position = class_position
        self.k = k
        self.has_outlier = has_outlier
        self.cam_num = cam_num
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (image_path_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        id_dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]   # from which index to obtain the label
            if class_index not in list(id_dict.keys()):
                id_dict[class_index] = [index]
            else:
                id_dict[class_index].append(index)
        return id_dict

    def _generate_list(self, id_dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''
        sample_list = []

        dict_copy = id_dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        outlier_cnt = 0
        for key in keys:
            value = dict_copy[key]
            if self.has_outlier and len(value)<=self.cam_num:
                random.shuffle(value)
                sample_list.append(value[0])  # sample outlier only one time
                outlier_cnt += 1
            elif len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k    # copy a person's image list for k-times
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
        if outlier_cnt > 0:
            print('in Sampler: outlier number= {}'.format(outlier_cnt))
        return sample_list



class ClassAndCameraBalancedSampler(Sampler):
    def __init__(self, data_source, num_instances=4, class_position=1):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        # for index, (_, pid, cam) in enumerate(data_source):
        for index, each_input in enumerate(data_source):
            pid = each_input[class_position]
            cam = each_input[2]
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for ii in indices:
            curr_id = self.pids[ii]
            indexes = np.array(self.pid_index[curr_id])
            cams = np.array(self.pid_cam[curr_id])
            uniq_cams = np.unique(cams)
            if len(uniq_cams) >= self.num_instances:  # more cameras than per-class-instances
                sel_cams = np.random.choice(uniq_cams, size=self.num_instances, replace=False)
                for cc in sel_cams:
                    ind = np.where(cams==cc)[0]
                    sel_idx = np.random.choice(indexes[ind], size=1, replace=False)
                    ret.append(sel_idx[0])
            else:
                sel_cams = np.random.choice(uniq_cams, size=self.num_instances, replace=True)
                for cc in np.unique(sel_cams):
                    sample_num = len(np.where(sel_cams == cc)[0])
                    ind = np.where(cams == cc)[0]
                    if len(ind) >= sample_num:
                        sel_idx = np.random.choice(indexes[ind], size=sample_num, replace=False)
                    else:
                        sel_idx = np.random.choice(indexes[ind], size=sample_num, replace=True)
                    for idx in sel_idx:
                        ret.append(idx)
        return iter(ret)

class ClusterProxyBalancedSampler(Sampler):
    '''
    Cluster-proxy balanced sampler. Samples are equally collected from different proxies in different clusters.

    Steps:
    1. Randomly select a cluster `c_i` from all clusters. Add it into the selected set.
    2. Randomly select a proxy `p_j` in the chosen cluster `c_i`.
    3. Randomly select `k` samples from `p_j` in `c_i`.
    4. Repeat until all `batchsize // num_instances` proxies are sampled.
    '''
    def __init__(self, samples, k=4, has_outlier=False, cam_num=0):

        self.samples = samples
        self.k = k
        self.has_outlier = has_outlier
        self.cam_num = cam_num
        self.dicts = self._tuple2dict(self.samples) # label -> img_index

    def __iter__(self):
        self.sample_list = self._generate_list(self.dicts)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (image_path_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        cluster2proxy_dict = {}
        proxy2id_dict = {}
        for index, each_input in enumerate(inputs):
            clbl = each_input[1]
            plbl = each_input[4]

            # Record cluster proxy mappings
            if clbl not in cluster2proxy_dict.keys():
                cluster2proxy_dict[clbl] = [plbl]
            else:
                cluster2proxy_dict[clbl].append(plbl)

            # Record proxy label mappings
            if plbl not in proxy2id_dict.keys():
                proxy2id_dict[plbl] = [index]
            else:
                proxy2id_dict[plbl].append(index)
        return cluster2proxy_dict, proxy2id_dict

    def _generate_list(self, dicts: List[dict]):
        '''
        dicts: list of dicts. containing cluster2id and proxy2id.
        '''
        sample_list = []
        cluster2proxy_dict, proxy2id_dict = dicts
        
        # Check each cluster for proxies
        cluster2proxy_dict_copy = cluster2proxy_dict.copy()
        clusters = list(cluster2proxy_dict_copy.keys())
        random.shuffle(clusters)
        for c in clusters:
            proxies = cluster2proxy_dict_copy[c]
            sel_proxy = random.sample(proxies, k=1)[0]
            img_indices = proxy2id_dict[sel_proxy]
            if len(img_indices) >= self.k:
                random.shuffle(img_indices)
                sample_list.extend(img_indices[:self.k])
            else:
                img_indices = img_indices * self.k
                random.shuffle(img_indices)
                sample_list.extend(img_indices[:self.k])
        return sample_list
    
class HardProxyBalancedSampler(Sampler):
    """
    对proxy进行PK均衡采样，每个proxy内选择K个距离proxy中心最远的样本。
    """
    pass