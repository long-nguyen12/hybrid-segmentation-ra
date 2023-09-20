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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from semseg.models import *
from semseg.datasets.colondb import * 
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate
import os
from torchviz import make_dot
import torch.nn.functional as F
from semseg.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

batch_size = 4

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def main(cfg, gpu, save_dir, train_loader, val_loader):
    start = time.time()
    best_mIoU = 0.0
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], 2)
    model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters {model_cfg['NAME']}: {total_params}")

    writer = SummaryWriter(str(save_dir / 'logs'))
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25]

    iters_per_epoch = len(train_loader.dataset) // batch_size

    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

   
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (img, lbl) in pbar:
            for rate in size_rates:
                optimizer.zero_grad(set_to_none=True)
                
                img = Variable(img).cuda()
                lbl = Variable(lbl).cuda()

                trainsize = int(352*rate)
                if rate != 1:
                    if lbl.shape[1] != 1:
                        lbl = lbl.unsqueeze(1).float()
                    img = F.interpolate(img, size=(trainsize, trainsize), mode='bicubic', align_corners=True)
                    lbl = F.interpolate(lbl, size=(trainsize, trainsize), mode='bicubic', align_corners=True)

                logits, score4, score3, score2, score1 = model(img)
                loss_0 = structure_loss(logits, lbl)
                loss_4 = structure_loss(score4, lbl)
                loss_3 = structure_loss(score3, lbl)
                loss_2 = structure_loss(score2, lbl)
                loss_1 = structure_loss(score1, lbl)
                loss = loss_0 + loss_2 + loss_3 + loss_4 + loss_1
                
                loss.backward()

                if rate == 1:
                    loss_record.update(loss.data, train_cfg['BATCH_SIZE'])
                # loss_record.update(loss.data, train_cfg['BATCH_SIZE'])

                optimizer.step()

            # if iter % 50 == 0 or iter == iters_per_epoch:
            #     print('{} Epoch [{:03d}/{:03d}], Step [{:03d}/{:03d}], '
            #         '[loss: {:.4f}], [lr: {:.9f}]'.
            #         format(datetime.now(), epoch, epochs, iter, iters_per_epoch,
            #                 loss_record.show(), optimizer.param_groups[0]['lr']))

        # if (epoch + 1) >= 10 and (epoch + 1) % 5 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1

            scheduler.step()
            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.data

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")

        train_loss /= iter+1
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
            
        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            miou = evaluate(model, val_loader, device, "Training")[0]
            
            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.state_dict(), save_dir / "best.pth")
            torch.save(model.state_dict(), save_dir / f"checkpoint{epoch+1}.pth")

            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")
        
    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)
    print(time.strftime("%H:%M:%S", end))

class PolypDB(Dataset):
    def __init__(self, root: str, transform = None) -> None:
        super().__init__()
        self.transform = transform
        self.n_classes = 2
        self.ignore_label = -1

        img_path = Path(root) / 'images' 
        self.files = list(img_path.glob('*.png'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} images.")

    @staticmethod
    def convert_to_mask(mask):
        h, w = mask.shape[:2]
        seg_mask = np.zeros((h,w, len(PALETTE)))
        for i, label in enumerate(PALETTE):
            seg_mask[:, :, i] = np.all(mask == label, axis=-1)
        return seg_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'masks')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(lbl_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert_to_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]
            mask = transformed["mask"]
            return image.float(), mask.argmax(dim=2).long()
        
        else:
            return image.float(), mask.argmax(dim=2).long()

def create_dataloaders(dir, image_size, batch_size, num_workers=os.cpu_count()):
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        # A.RandomScale(scale_limit=(0.5, 2), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    dataset = PolypDB(root=dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True, 
            pin_memory=True
        )
    
    return dataloader, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)

    dataloader, dataset = create_dataloaders('data/data/TrainDataset/', [352, 352], batch_size)

    train_ratio = 0.8
    val_ratio = 0.2
    num_samples = len(dataloader.dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_val_samples = num_samples - num_train_samples
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train_samples, num_val_samples])

    # sub_samples = len(train_set.dataset)
    # sub_train_samples = int(0.1 * sub_samples)
    # sub_set, left_set = torch.utils.data.random_split(dataset, [sub_train_samples, sub_samples - sub_train_samples])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)   

    main(cfg, gpu, save_dir, train_loader, val_loader)
    cleanup_ddp()