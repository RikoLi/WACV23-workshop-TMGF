'''
Memory bank loss implementations.
Implementations are inspired by SpCL and O2CAP, thanks for their excellent works! 
SpCL: https://github.com/yxgeee/SpCL
O2CAP: https://github.com/Terminator8758/O2CAP
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda import amp

class PartMatMul(Function):
    """
    Matrix multiplication with memory bank update. An extra part dim is added.
    In forwarding, it only applies a matmul operation between anchors and memory bank.
    In backwarding, it update the memory bank with momentum.
    """
    
    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, em, alpha):
        ctx.em = em
        ctx.alpha = alpha
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.matmul(ctx.em.permute(0,2,1)) # (n_part, b, c) x (n_part, c, n_proxy) -> (n_part, b, n_proxy)
        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.matmul(ctx.em) # (n_part, b, c)
        
        for i, y in enumerate(targets):
            x = inputs[:,i,:] # (n_part, c)
            ctx.em[:,y,:] = ctx.alpha * ctx.em[:,y,:] + (1.0 - ctx.alpha) * x
            ctx.em[:,y,:] /= ctx.em[:,y,:].norm(dim=1).unsqueeze(-1)
        
        return grad_inputs, None, None, None

def part_matmul(inputs, targets, em, alpha):
    return PartMatMul.apply(inputs, targets, em, alpha)

class MultiPartMemory(nn.Module):
    def __init__(self, cfg):
        """
        Multi-part offline/online loss with momentum proxy memory bank.
        
        Params:
            cfg: Config instance.
        Returns:
            A MultiPartMemory instance.
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = cfg.MEMORY_BANK.PROXY_TEMP
        self.momentum = cfg.MEMORY_BANK.MOMENTUM
        self.neg_k = cfg.MEMORY_BANK.BG_KNN
        self.posK = cfg.MEMORY_BANK.POS_K
        self.balance_w = cfg.MEMORY_BANK.BALANCE_W
        self.part_weight = cfg.MEMORY_BANK.PART_W
        self.num_parts = cfg.MODEL.NUM_PARTS
        
        self.all_proxy_labels = None
        self.proxy_memory = None
        self.proxy2cluster = None
        self.cluster2proxy = None
        self.part_proxies = None
        self.unique_cams = None
        self.cam2proxy = None
        
    def forward(self, feature_dict, targets, cam=None, epoch=None):
        
        # proxy labels in a batch
        batch_proxy_labels = self.all_proxy_labels[targets].to(self.device)
        
        # loss computation
        all_feats = torch.cat([feature_dict['global'].unsqueeze(0), feature_dict['part']], dim=0)
        all_scores = part_matmul(all_feats, batch_proxy_labels, self.proxy_memory, self.momentum) # (n_part, b, n_proxy)
        all_scaled_scores = all_scores / self.temp
        global_off_loss, part_off_loss = self.offline_loss_part_parallel(all_scaled_scores, batch_proxy_labels)
        part_off_loss = part_off_loss.mean()
        
        all_temp_scores = all_scores.detach().clone()
        global_on_loss, part_on_loss = self.online_loss_part_parallel(all_scaled_scores, batch_proxy_labels, all_temp_scores)
        part_on_loss = part_on_loss.mean()
        
        # part loss weight
        part_off_loss *= self.part_weight
        part_on_loss *= self.part_weight
        
        loss_dict = {
            'loss': global_off_loss + global_on_loss + part_off_loss + part_on_loss,
            'global_off_loss': global_off_loss,
            'global_on_loss': global_on_loss,
            'part_off_loss': part_off_loss,
            'part_on_loss': part_on_loss
        }
        return loss_dict
    
    def offline_loss_part_parallel(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute offline loss for both global and part level.
        All parts are handled parallelly to alleviate time consumption.
        
        Params:
            scores: Scaled batch-proxy similarity scores.
            labels: Proxy labels in a batch.
        Returns:
            Offline global and part losses.
        """
        
        temp_scores = scores.detach().clone()
        loss = 0
        
        if scores.size(0) > 1:
            part_loss = 0
        else:
            part_loss = torch.tensor(0).type_as(scores)
        
        for i in range(scores.size(1)):
            pos_ind = torch.tensor(self.cluster2proxy[self.proxy2cluster[labels[i].item()].item()]).type_as(labels)
            temp_scores[:, i, pos_ind] = 10000.0
            sel_ind = torch.argsort(temp_scores[:, i, :])[:, -self.neg_k-len(pos_ind):] # (n_part, neg_k+pos_k)
            sel_input = scores[:, i, :].gather(dim=1, index=sel_ind) # (n_part, neg_k+pos_k)
            sel_target = torch.zeros(sel_input.shape).type_as(sel_input) # (n_part, neg_k+pos_k)
            sel_target[:, -len(pos_ind):] = 1.0 / len(pos_ind)
            logit = -1.0 * (F.log_softmax(sel_input, dim=1) * sel_target) # (n_part, neg_k+pos_k)
            loss += logit[0,:].sum()
            
            # compute part loss when there exists part feature
            if scores.size(0) > 1:
                part_loss += logit[1:,:].sum(dim=1)
        
        loss /= scores.size(1)
        part_loss /= scores.size(1)
        
        return loss, part_loss

    def online_loss_part_parallel(self, scores: torch.Tensor, labels: torch.Tensor, temp_scores: torch.Tensor):
        """
        Compute online loss for both global and part level.
        All parts and batch samples are handled parallelly to alleviate time consumption.
        
        Params:
            scores: Scaled batch-proxy similarity scores.
            labels: Proxy labels in a batch.
            temp_scores: Detached scores for positive/negative samples retrieval.
        Returns:
            Online global and part losses.
        """
        # compute online similarity
        temp_memory = self.proxy_memory.detach().clone()
        proxy_sims = torch.matmul(temp_memory, temp_memory.permute(0,2,1)) # (1+N_part, N_proxy, N_proxy)
        sims = self.balance_w * temp_scores + (1 - self.balance_w) * proxy_sims[:, labels, :] # (1+N_part, B, N_proxy)
        
        # CA-NN: camera-aware nearest neighbors
        all_cam_tops = []
        for cc in self.unique_cams:
            proxy_inds = self.cam2proxy[int(cc)].long().to(self.device) # 当前相机下的proxy label
            max_idx = sims[:, :, proxy_inds].argmax(dim=2)
            all_cam_tops.append(proxy_inds[max_idx])
            
        # retrieve positive samples
        all_cam_tops = torch.stack(all_cam_tops, dim=-1) # (1+N_part, B, N_cam)
        top_sims = torch.gather(sims, dim=2, index=all_cam_tops) # (1+N_part, B, N_cam)
        sel_inds = torch.argsort(top_sims, dim=2)[:, :, -self.posK:]
        pos_inds = torch.gather(all_cam_tops, dim=2, index=sel_inds)
        scatter_sims = torch.scatter(sims, dim=2, index=pos_inds, src=10000.0*torch.ones(pos_inds.shape).type_as(sims)) # (1+N_part, B, N_proxy)
        top_inds = torch.sort(scatter_sims, dim=2)[1][:, :, -self.neg_k-self.posK:] # (1+N_part, B, N_pn)
        sel_inputs = torch.gather(scores, dim=2, index=top_inds)
        sel_targets = torch.zeros(sel_inputs.shape).type_as(sel_inputs)
        sel_targets[:, :, -self.posK:] = 1.0 / self.posK
        
        # global loss
        loss = -1.0 * (F.log_softmax(sel_inputs[0], dim=1) * sel_targets[0]).sum(dim=1).mean() # scalar
        
        # part loss
        if scores.size(0) > 1:
            part_loss = -1.0 * (F.log_softmax(sel_inputs[1:], dim=2) * sel_targets[1:]).sum(dim=2).mean(dim=1) # (N_part, )
        else:
            part_loss = torch.tensor(0).type_as(loss)
        
        return loss, part_loss
        