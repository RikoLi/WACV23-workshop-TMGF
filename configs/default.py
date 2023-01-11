from yacs.config import CfgNode as CN

_C = CN()

# Random seed
_C.SEED = 1

# Default task name
_C.TASK_NAME = 'untitiled_task'

# Task description
_C.DESC = 'default_desc'

# Model settings
_C.MODEL = CN()
_C.MODEL.ARCH = 'tmgf'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.SIE_COEF = 3.0
_C.MODEL.SIE_CAMERA = 6
_C.MODEL.SIE_VIEW = 0
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATTN_DROP_RATE = 0.0
_C.MODEL.PRETRAIN_HW_RATIO = 1
_C.MODEL.GEM_POOL = False
_C.MODEL.STEM_CONV = False
_C.MODEL.PRETRAIN_PATH = '/home/ljc/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
_C.MODEL.NUM_PARTS = 5                  # total number of parts
_C.MODEL.HAS_HEAD = True                # whether to use muti-grained projection heads
_C.MODEL.HAS_EARLY_FEATURE = True       # whether to obtain (L-1)-th layer output feature
_C.MODEL.ENABLE_EARLY_NORM = False      # whether to apply LayerNorm on (L-1)-th layer output feature
_C.MODEL.GLOBAL_FEATURE_TYPE = 'mean'   # which global token fusion method to use: mean, b1, b2
_C.MODEL.GRANULARITIES = [2, 3]         # number of part splits in each branch, sum up to MODEL.NUM_PART
_C.MODEL.BRANCH = 'all'                 # which branch to use: all, b1, b2


# Dataset settings
_C.DATASET = CN()
_C.DATASET.ROOT_DIR = '/home/ljc/datasets'
_C.DATASET.NAME = 'Market1501'

# Sampler settings
_C.SAMPLER = CN()
_C.SAMPLER.TYPE = 'proxy_balance'
_C.SAMPLER.NUM_INSTANCES = 4

# Clustering settings
_C.CLUSTER = CN()
_C.CLUSTER.EPS = 0.5
_C.CLUSTER.MIN_SAMPLES = 4
_C.CLUSTER.K1 = 20
_C.CLUSTER.K2 = 6

# Input settings
_C.INPUT = CN()
_C.INPUT.HEIGHT = 384
_C.INPUT.WIDTH = 128
_C.INPUT.PIXEL_MEAN = [0.3525, 0.3106, 0.3140]  # LUPerson statistics
_C.INPUT.PIXEL_STD = [0.2660, 0.2522, 0.2505]   # LUPerson statistics

# Optimizer settings
_C.OPTIM = CN()
_C.OPTIM.NAME = 'SGD'
_C.OPTIM.BASE_LR = 3.5e-4
_C.OPTIM.WEIGHT_DECAY = 0.0005
_C.OPTIM.WEIGHT_DECAY_BIAS = 0.0005
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.BIAS_LR_FACTOR = 1.0
_C.OPTIM.SCHEDULER_TYPE = 'warmup'
_C.OPTIM.WARMUP_EPOCHS = 10
_C.OPTIM.WARMUP_FACTOR = 0.01
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.WARMUP_METHOD = 'linear'
_C.OPTIM.MILESTONES = [20, 40]

# Logging settings
_C.LOG = CN()
_C.LOG.PRINT_FREQ = 50
_C.LOG.LOG_DIR = '/home/ljc/works/TMGF/logs'
_C.LOG.CHECKPOINT = CN()
_C.LOG.CHECKPOINT.SAVE_DIR = '/home/ljc/works/TMGF/ckpt'
_C.LOG.CHECKPOINT.SAVE_INTERVAL = 100
_C.LOG.CHECKPOINT.LOAD_DIR = ''
_C.LOG.CHECKPOINT.LOAD_EPOCH = 0
_C.LOG.SAVE_BENCHMARK = False
_C.LOG.BENCHMARK_PATH = '/home/ljc/works/TMGF/benchmark.csv'

# Training settings
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.ITERS = 200
_C.TRAIN.BATCHSIZE = 32
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.FP16 = False

# Test settings
_C.TEST = CN()
_C.TEST.BATCHSIZE = 32
_C.TEST.NUM_WORKERS = 8
_C.TEST.EVAL_STEP = 10
_C.TEST.RE_RANK = False

# Memory bank settings
_C.MEMORY_BANK = CN()
_C.MEMORY_BANK.MOMENTUM = 0.2
_C.MEMORY_BANK.PROXY_TEMP = 0.07
_C.MEMORY_BANK.BG_KNN = 50
_C.MEMORY_BANK.POS_K = 3
_C.MEMORY_BANK.BALANCE_W = 0.15
_C.MEMORY_BANK.PART_W = 1.0

def get_cfg_defaults():
    return _C.clone()