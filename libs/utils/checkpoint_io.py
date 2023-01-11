import os
import os.path as osp
import torch

def save_checkpoint(model, optimizer, scheduler, ckpt_save_dir, epoch):
    if not osp.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir, exist_ok=True)
    torch.save(model.state_dict(), osp.join(ckpt_save_dir, 'weight_{}.pth'.format(epoch)))
    torch.save(optimizer.state_dict(), osp.join(ckpt_save_dir, 'optim_{}.pth'.format(epoch)))
    torch.save(scheduler.state_dict(), osp.join(ckpt_save_dir, 'scheduler_{}.pth'.format(epoch)))

def load_checkpoint(model, optimizer, scheduler, ckpt_load_dir, ckpt_load_ep):
    weight = torch.load(osp.join(ckpt_load_dir, 'weight_{}.pth').format(ckpt_load_ep))
    opt_params = torch.load(osp.join(ckpt_load_dir, 'optim_{}.pth'.format(ckpt_load_ep)))
    sch_params = torch.load(osp.join(ckpt_load_dir, 'scheduler_{}.pth'.format(ckpt_load_ep)))
    
    model.load_state_dict(weight)
    optimizer.load_state_dict(opt_params)
    scheduler.load_state_dict(sch_params)

    return model, optimizer, scheduler