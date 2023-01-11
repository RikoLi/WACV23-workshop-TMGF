import time
from torch.cuda import amp
from .utils.meters import AverageMeter

class _BaseTrainer:
    """The most basic trainer class."""
    def __init__(self, encoder, memory) -> None:
        super().__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        raise NotImplementedError

    def _parse_data(self, inputs):
        imgs, _, _, cams, index_target, _ = inputs   # img, fname, pseudo_label, camid, img_index, accum_label
        return imgs.cuda(), cams.cuda(), index_target.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
    
class ViTTrainerFp16(_BaseTrainer):
    """
    ViT trainer with FP16 forwarding.
    """
    def __init__(self, encoder, memory) -> None:
        super().__init__(encoder, memory)
        
    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, fp16=False):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        
        # amp fp16 training
        scaler = amp.GradScaler() if fp16 else None
        
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, cams, index_target = self._parse_data(inputs)

            # loss
            with amp.autocast(enabled=fp16):
                # forward
                f_out = self._forward(inputs, cam_label=cams) # dict: global & part features

                # compute loss with the memory
                loss_dict = self.memory(f_out, index_target, cams, epoch)
                loss = loss_dict['loss']
                
            optimizer.zero_grad()
            
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}] '
                      'Time: {:.3f} ({:.3f}), '
                      'Data: {:.3f} ({:.3f}), '
                      'Loss: {:.3f} ({:.3f}), '
                      '{}'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              ', '.join(['{}: {:.3f}'.format(k, v) for k, v in loss_dict.items()])))

    def _parse_data(self, inputs):
        imgs, _, _, cams, index_target, _ = inputs   # img, fname, pseudo_label, camid, img_index, accum_label
        return imgs.cuda(), cams.cuda(), index_target.cuda()

    def _forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
    