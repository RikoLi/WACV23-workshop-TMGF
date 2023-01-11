import torch

def make_vit_optimizer(cfg, model):
    """
    Create ViT optimizer.
    
    Params:
        cfg: Config instance.
        model: The model to be optimized.
    Returns:
        An optimizer.
    """
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.OPTIM.BASE_LR
        weight_decay = cfg.OPTIM.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.BIAS_LR_FACTOR
            weight_decay = cfg.OPTIM.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.OPTIM.NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.OPTIM.NAME)(params, momentum=cfg.OPTIM.MOMENTUM)
    elif cfg.OPTIM.NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.BASE_LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.OPTIM.NAME)(params)

    return optimizer