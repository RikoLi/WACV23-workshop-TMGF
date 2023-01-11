import os.path as osp
import sys
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import time
from configs.default import get_cfg_defaults
from datetime import timedelta
from libs.utils.prepare_data import get_data, get_test_loader, get_train_loader
from libs.utils.prepare_model import create_vit_model
from libs import trainers
from libs.models import mb
from libs.evaluators import Evaluator, extract_multipart_vit_features, save_benchmark
from libs.utils.logging import Logger
from libs.utils.checkpoint_io import load_checkpoint, save_checkpoint
from libs.utils.prepare_optimizer import make_vit_optimizer
from libs.utils.prepare_scheduler import create_scheduler
from libs.utils.clustering import dbscan_clustering, cam_label_split, get_centers


def main(cfg):
    # Check output dir
    assert osp.exists(cfg.LOG.LOG_DIR)
    assert osp.exists(cfg.LOG.CHECKPOINT.SAVE_DIR)

    start_time = time.monotonic()
    
    # Build task folder
    task_name = time.strftime('%Y%m%d') + '_' + cfg.TASK_NAME
    log_file_name = osp.join(cfg.LOG.LOG_DIR, task_name+'.txt')
    ckpt_save_dir = osp.join(cfg.LOG.CHECKPOINT.SAVE_DIR, task_name)


    # Print settings
    sys.stdout = Logger(log_file_name)
    print("==========\n{}\n==========".format(cfg))
    print('=> Task name:', task_name)
    print('=> Description:', cfg.DESC)


    # Create datasets
    iters = cfg.TRAIN.ITERS if (cfg.TRAIN.ITERS>0) else None
    print("=> Load unlabeled dataset")
    dataset = get_data(cfg.DATASET.NAME, cfg.DATASET.ROOT_DIR)

    # Create model
    model = create_vit_model(cfg)

    # Create memory
    memory = mb.MultiPartMemory(cfg).cuda()
    
    # Get dataloaders
    cluster_loader = get_test_loader(cfg, dataset, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.TEST.BATCHSIZE, cfg.TEST.NUM_WORKERS, testset=sorted(dataset.train))
    test_loader = get_test_loader(cfg, dataset, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.TEST.BATCHSIZE, cfg.TEST.NUM_WORKERS)
    
    # Evaluator
    evaluator = Evaluator(cfg, model)


    # Optimizer & scheduler
    optimizer = make_vit_optimizer(cfg, model)
    lr_scheduler = create_scheduler(cfg, optimizer)
    
    # Load checkpoint
    if len(cfg.LOG.CHECKPOINT.LOAD_DIR) == 0:
        start_ep = 0 # default: start from beginning
        print('=> Train from beginning.')
    else:
        model, optimizer, lr_scheduler = load_checkpoint(model, optimizer, lr_scheduler, cfg.LOG.CHECKPOINT.LOAD_DIR, cfg.LOG.CHECKPOINT.LOAD_EPOCH)
        start_ep = cfg.LOG.CHECKPOINT.LOAD_EPOCH
        print('=> Continue training from epoch={}, load checkpoint from {}'.format(start_ep, cfg.LOG.CHECKPOINT.LOAD_DIR))

    # Trainer
    trainer = trainers.ViTTrainerFp16(model, memory)

    # Training pipeline
    for epoch in range(start_ep, cfg.TRAIN.EPOCHS):
        print('=> EPOCH num={}'.format(epoch+1))

        # Feature extraction
        print('=> Extract features...')
        features, part_feats, _ = extract_multipart_vit_features(model, cluster_loader, cfg.MODEL.NUM_PARTS)
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        part_feats = [torch.cat([pf[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0) for pf in part_feats]
        
        # Clustering for pseudo labels
        cluster_labels = dbscan_clustering(cfg, features)
        
        
        # Camera proxy generation
        print('=> cam-split with global features')
        all_img_cams = np.array([c for _, _, c in sorted(dataset.train)])
        proxy_labels = cam_label_split(cluster_labels, all_img_cams)

        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_proxies = len(set(proxy_labels)) - (1 if -1 in proxy_labels else 0)
        num_outliers = len(np.where(proxy_labels == -1)[0])
        print('=> Global feature clusters: {}\n=> Generated proxies: {}\n=> Outliers: {}'.format(
            num_clusters, num_proxies, num_outliers
        ))
        
        
        # Add pseudo labels into training set
        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), gcl, pl) in enumerate(zip(sorted(dataset.train), cluster_labels, proxy_labels)):
            if gcl != -1 and pl != -1:
                pseudo_labeled_dataset.append((fname, gcl, cid, i, pl))


        # Cluster-proxy mappings
        proxy_labels = torch.from_numpy(proxy_labels).long()
        cluster_labels = torch.from_numpy(cluster_labels).long()
        cluster2proxy = {} # global cluster label -> proxy
        proxy2cluster = {} # proxy -> global cluster label
        cam2proxy = {} # cam -> proxy
        for p in range(0, int(proxy_labels.max() + 1)):
            proxy2cluster[p] = torch.unique(cluster_labels[proxy_labels == p])
        for c in range(0, int(cluster_labels.max() + 1)):
            cluster2proxy[c] = torch.unique(proxy_labels[cluster_labels == c])
        for cc in range(0, int(all_img_cams.max() + 1)):
            cam2proxy[cc] = torch.unique(proxy_labels[all_img_cams == cc])
            cam2proxy[cc] = cam2proxy[cc][cam2proxy[cc] != -1] # remove outliers

        # Set memory attributes
        memory.all_proxy_labels = proxy_labels # proxy label of all samples
        memory.proxy2cluster = proxy2cluster
        memory.cluster2proxy = cluster2proxy
        
        # Stack into a single memory
        proxy_memory = [get_centers(features.numpy(), proxy_labels.numpy()).cuda()] + \
            [get_centers(f.numpy(), proxy_labels.numpy()).cuda() for f in part_feats]
        memory.proxy_memory = torch.stack(proxy_memory, dim=0) # (n_part, n_proxy, c)
        
        
        # camera-proxy mapping
        memory.unique_cams = torch.unique(torch.from_numpy(all_img_cams))
        memory.cam2proxy = cam2proxy
                
        # Get a train loader
        train_loader = get_train_loader(cfg, dataset, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH,
                                        cfg.TRAIN.BATCHSIZE, cfg.TRAIN.NUM_WORKERS,
                                        cfg.SAMPLER.NUM_INSTANCES, iters, 
                                        trainset=pseudo_labeled_dataset)


        
        # Train one epoch
        curr_lr = lr_scheduler._get_lr(epoch+1)[0] if cfg.OPTIM.SCHEDULER_TYPE == 'cosine' else lr_scheduler.get_lr()[0]
        print('=> Current Lr: {:.2e}'.format(curr_lr))
        train_loader.new_epoch()
        trainer.train(epoch+1, train_loader, optimizer, print_freq=cfg.LOG.PRINT_FREQ, train_iters=len(train_loader), fp16=cfg.TRAIN.FP16)

        # Update scheduler
        if cfg.OPTIM.SCHEDULER_TYPE == 'cosine':
            lr_scheduler.step(epoch+1)
        else:
            lr_scheduler.step()

        # Save checkpoint
        if (epoch+1) % cfg.LOG.CHECKPOINT.SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, lr_scheduler, ckpt_save_dir, epoch+1)
            print('=> Checkpoint is saved.')

        # Evaluation
        if ((epoch+1) % cfg.TEST.EVAL_STEP == 0 or (epoch == cfg.TRAIN.EPOCHS - 1)):
            print('=> Epoch {} test: '.format(epoch+1))
            cmc, mAP = evaluator.evaluate_vit(test_loader, dataset.query, dataset.gallery, cmc_flag=True, rerank=cfg.TEST.RE_RANK)
        
        torch.cuda.empty_cache()
        print('=> CUDA cache is released.')
        print('')


    end_time = time.monotonic()
    dtime = timedelta(seconds=end_time - start_time)
    print('=> Task finished: {}'.format(task_name))
    print('Total running time: {}'.format(dtime))
    
    # Save benchmark
    if cfg.LOG.SAVE_BENCHMARK:
        save_benchmark(cfg, mAP, cmc, task_name, dtime)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='', help='Config file path.')
    parser.add_argument('opts', help='Modify config options using CMD.', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Load config using yacs
    cfg = get_cfg_defaults()
    if args.conf != '':
        cfg.merge_from_file(args.conf)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Init env.
    if cfg.SEED is not None:
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Run
    main(cfg)
