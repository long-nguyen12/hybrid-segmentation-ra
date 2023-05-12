from semseg.datasets.cvc import create_dataloaders
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate
import os
from torchviz import make_dot

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']

    trainloader, trainset = create_dataloaders(dataset_cfg['ROOT'], 'train', train_cfg['IMAGE_SIZE'], train_cfg['BATCH_SIZE'], num_workers)
    valloader, valset = create_dataloaders(dataset_cfg['ROOT'], 'val', eval_cfg['IMAGE_SIZE'], train_cfg['BATCH_SIZE'], num_workers)
    
    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], 2)
    
    # model.init_pretrained(model_cfg['PRETRAINED'])

    model = model.to(device)
    # return
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (img, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)
            img = img.to(device)
            lbl = lbl.to(device)

            with autocast(enabled=train_cfg['AMP']):
                # logits = model(img)
                # loss = loss_fn(logits, lbl)
                logits, score4, score3, score2 = model(img)
                loss_0 = loss_fn(logits, lbl)
                loss_1 = loss_fn(score2, lbl)
                loss_2 = loss_fn(score3, lbl)
                loss_3 = loss_fn(score4, lbl)
                loss = loss_0 + loss_1 + loss_2 + loss_3
                loss = loss / train_cfg['BATCH_SIZE']
                
                loss.backward()

            scaler.scale(loss)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            miou = evaluate(model, valloader, device)[-1]
            writer.add_scalar('val/mIoU', miou, epoch)
            
            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/kvasir.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()