import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from .faiss_rerank import compute_jaccard_distance


def cam_label_split(cluster_labels, all_img_cams):
    """
    Split proxies using camera labels.
    
    Params:
        cluster_labels: Pseudo labels from DBSCAN clustering.
        all_img_cams: Camera labels of all images.
    Returns:
        Proxy labels of all images.
    """
    proxy_labels = -1 * np.ones(cluster_labels.shape, cluster_labels.dtype)
    cnt = 0
    for i in range(0, int(cluster_labels.max() + 1)):
        inds = np.where(cluster_labels == i)[0]
        local_cams = all_img_cams[inds]
        for cc in np.unique(local_cams):
            pc_inds = np.where(local_cams == cc)[0]
            proxy_labels[inds[pc_inds]] = cnt
            cnt += 1
    return proxy_labels

def dbscan_clustering(cfg, features):
    """
    DBSCAN clustering. Generate pseudo labels.
    
    Params:
        cfg: Config instance.
        features: Image features extracted by the model.
    Returns:
        Pseudo cluster labels of all images.
    """
    
    rerank_dist = compute_jaccard_distance(features, k1=cfg.CLUSTER.K1, k2=cfg.CLUSTER.K2)
    print('=> Global DBSCAN params: eps={:.3f}, min_samples={:.3f}'.format(cfg.CLUSTER.EPS, cfg.CLUSTER.MIN_SAMPLES))
    
    dbscan = DBSCAN(eps=cfg.CLUSTER.EPS, min_samples=cfg.CLUSTER.MIN_SAMPLES, metric='precomputed', n_jobs=-1)
    cluster_labels = dbscan.fit_predict(rerank_dist)
    
    return cluster_labels

def get_centers(features, labels):
    """
    Get L2-normalized centers of all pseudo classes.
    
    Params:
        features: Image features extracted by the model.
        labels: Pseudo labels of all features.
    Returns:
        L2-normalized centers of all pseudo classes.
    """
    num_ids = len(set(labels)) - (1 if -1 in labels else 0)
    centers = np.zeros((num_ids, features.shape[1]), dtype=np.float32)
    for i in range(num_ids):
        idx = torch.where(torch.from_numpy(labels) == i)[0].numpy()
        temp = features[idx,:]
        if len(temp.shape) == 1:
            temp = temp.reshape(1, -1)
        centers[i,:] = temp.mean(0)
    centers = torch.from_numpy(centers)
    return F.normalize(centers, dim=1)
