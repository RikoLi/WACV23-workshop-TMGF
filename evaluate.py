import argparse
import random
import torch
import numpy as np
import time
from datetime import timedelta
from configs.default import get_cfg_defaults
from libs.utils.prepare_model import create_vit_model
from libs.utils.prepare_data import get_data, get_test_loader
from libs.evaluators import Evaluator

def evaluate(cfg, weight_path):
    model = create_vit_model(cfg)
    dataset = get_data(cfg.DATASET.NAME, cfg.DATASET.ROOT_DIR)
    test_loader = get_test_loader(cfg, dataset, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.TEST.BATCHSIZE, cfg.TEST.NUM_WORKERS)
    evaluator = Evaluator(cfg, model)
    
    weight = torch.load(weight_path)
    model.load_state_dict(weight)
    print('=> Model weights loaded.')
    
    print('=> Start evaluation...')
    st = time.monotonic()
    evaluator.evaluate_vit(test_loader, dataset.query, dataset.gallery, cmc_flag=True, rerank=cfg.TEST.RE_RANK)
    et = time.monotonic()
    dt = timedelta(seconds=et-st)
    print('=> Evaluation time: {}'.format(dt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='', help='Config file path.')
    parser.add_argument('--weight', type=str, default='', help='Model parameter weight (.pth format) path.')
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
    evaluate(cfg, args.weight)