import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../..')))

from torch.utils.data import DataLoader
from libs import datasets
from .data import transforms as T
from .data import IterLoader
from .data.sampler import ClassUniformlySampler, RandomMultipleGallerySampler, ClusterProxyBalancedSampler
from .data.preprocessor import Preprocessor, CameraAwarePreprocessor

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print('root path= {}'.format(root))
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(cfg, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):
    # Preprocessing
    normalizer = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                             std=cfg.INPUT.PIXEL_STD)
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=cfg.INPUT.PIXEL_MEAN)
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)

    # Choose sampler type
    # class_position [1: cluster_label, 4: proxy_label]
    if cfg.SAMPLER.TYPE == 'proxy_balance':
        sampler = ClassUniformlySampler(train_set, class_position=4, k=num_instances)
    elif cfg.SAMPLER.TYPE == 'cluster_balance':
        sampler = ClassUniformlySampler(train_set, class_position=1, k=num_instances)
    elif cfg.SAMPLER.TYPE == 'cam_cluster_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=1, num_instances=num_instances)
    elif cfg.SAMPLER.TYPE == 'cam_proxy_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=4, num_instances=num_instances)
    elif cfg.SAMPLER.TYPE == 'cluster_proxy_balance':
        sampler = ClusterProxyBalancedSampler(train_set, k=num_instances)
    else:
        raise ValueError('Invalid sampler type name!')

    # Create dataloader
    train_loader = IterLoader(
                DataLoader(CameraAwarePreprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=False, pin_memory=True, drop_last=True), length=iters)
    return train_loader

def get_test_loader(cfg, dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                             std=cfg.INPUT.PIXEL_STD)

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader
