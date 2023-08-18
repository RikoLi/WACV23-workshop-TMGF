from torch import nn
from .. import models
         
def create_vit_model(cfg):
    """
    Create ViT model.
    
    Params:
        cfg: Config instance.
    Returns:
        The TMGF model.
    """
        
    model = models.create(cfg.MODEL.ARCH, arch=cfg.MODEL.ARCH,
                          img_size=[cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH], sie_coef=cfg.MODEL.SIE_COEF,
                          camera_num=cfg.MODEL.SIE_CAMERA, view_num=cfg.MODEL.SIE_VIEW,
                          stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                          drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
                          pretrain_path=cfg.MODEL.PRETRAIN_PATH, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO,
                          gem_pool=cfg.MODEL.GEM_POOL, stem_conv=cfg.MODEL.STEM_CONV, num_parts=cfg.MODEL.NUM_PARTS,
                          has_head=cfg.MODEL.HAS_HEAD, global_feature_type=cfg.MODEL.GLOBAL_FEATURE_TYPE,
                          granularities=cfg.MODEL.GRANULARITIES, branch=cfg.MODEL.BRANCH, has_early_feature=cfg.MODEL.HAS_EARLY_FEATURE,
                          enable_early_norm=cfg.MODEL.ENABLE_EARLY_NORM)
    model.cuda()
    return model
