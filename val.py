from semseg.datasets.cvc import create_dataloaders
import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from torchvision import utils
import os
from semseg.datasets.colondb import * 

@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(2, -1, device)

    save_preds = './result2/' + "/"
    save_labels = './labels2/' + "/"

    if not os.path.exists(save_labels):
        os.makedirs(save_labels)
    if not os.path.exists(save_preds):
        os.makedirs(save_preds)

    count = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        logits, score4, score3, score2, score1 = model(images)
        preds = torch.sigmoid(score1)
        # logits = model(images)
        # preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()

        # _pred = preds.to(torch.uint8)
        # _label = labels.to(torch.long)
        # utils.save_image(_label, 'labels.png')
        save_path = save_preds + str(count) + "_pred.png" 
        save_label_path = save_labels + str(count) + "_label.png" 
        count += 1
        utils.save_image(labels[0].float(), save_label_path)
        utils.save_image(preds, save_path)

        metrics.update(preds, labels)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou

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
        lbl_path = str(self.files[index]).replace('images', 'masks').replace('.png', '.png')
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

def create_dataloaders(dir, split, image_size, batch_size, num_workers=os.cpu_count()):
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
    if split == 'train':
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True, 
            pin_memory=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1, 
            pin_memory=True
        )
    
    return dataloader, dataset


def main(cfg, dataloader, dataset, _dataset):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], 2)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    acc, macc, dice, mdice, ious, miou = evaluate(model, dataloader, device)

    table = {
        'Class': list([
        'background',
        'polyp'
    ]) + ['Mean'],
        'IoU': ious + [miou],
        'Dice': dice + [mdice],
        'Acc': acc + [macc]
    }

    print(tabulate(table, headers='keys'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/kvasir.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    ds = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    for _dataset in ds:
        dataloader, dataset = create_dataloaders('data/data/TestDataset/' + _dataset, 'val', [352, 352], 1)
        main(cfg, dataloader, dataset, _dataset)