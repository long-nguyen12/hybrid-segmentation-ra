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
def evaluate(model, dataloader, device, folder):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(2, -1, device)

    save_preds = './result/preds/' + "/" + folder + "/"
    save_labels = './result/labels/' + "/" + folder + "/"
    save_images = './result/images/' + "/" + folder + "/"

    if not os.path.exists(save_labels):
        os.makedirs(save_labels)
    if not os.path.exists(save_preds):
        os.makedirs(save_preds)
    if not os.path.exists(save_images):
        os.makedirs(save_images)

    count = 0
    best_iou = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        logits, score4, score3, score2, score1 = model(images)
        preds = torch.sigmoid(score1)
        # logits = model(images)
        # preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()
        # current_iou = metrics.compute_single_iou(labels[0], preds)
        # if current_iou > best_iou:
        #     name = img_paths[0].split('/')[-1]
        #     save_path = save_preds + name + "_best.png" 
        #     save_label_path = save_labels + name + "_best.png" 
        #     count += 1
        #     utils.save_image(labels[0].float(), save_label_path)
        #     utils.save_image(preds, save_path)
        #     best_iou = current_iou
        #     print(best_iou, img_paths)

        metrics.update(preds, labels)
    
    miou = metrics.compute_mean_iou()
    mdice = metrics.compute_mean_dice()
    f1_score, mprecision, mrecall = metrics.compute_mean_f1_score()
    
    return miou, mdice, f1_score, mprecision, mrecall

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
        # A.HorizontalFlip(p=0.5),
        # A.RandomScale(scale_limit=(0.5, 2), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    dataset = PolypDB(root=dir, transform=transform)
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
    # print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], 2)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    miou, mdice, f1_score, mprecision, mrecall = evaluate(model, dataloader, device, _dataset)

    return miou, mdice, f1_score, mprecision, mrecall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    ds = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    # ds = ['ETIS-LaribPolypDB']

    for _dataset in ds:
        ious = []
        dices = []
        f1_scores = []
        precisions = []
        recalls = []
        print(_dataset)
        dataloader, dataset = create_dataloaders('data/data/TestDataset/' + _dataset, 'val', [352, 352], 1)
        for i in range(5):
            iou, dice, f1_score, mprecision, mrecall = main(cfg, dataloader, dataset, _dataset)
            ious.append(iou)
            dices.append(dice)
            f1_scores.append(f1_score)
            precisions.append(mprecision)
            recalls.append(mrecall)
            print(f'Mean IoU: {iou}, mean Dice: {dice}')

        ious = np.array(ious)
        dices = np.array(dices)
        f1_scores = np.array(f1_scores)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        print(f'Mean IoU: {np.mean(ious)}, mean Dice: {np.mean(dices)}, mean F1-Score: {round(np.mean(f1_scores), 3)}, mean Precision: {round(np.mean(precisions), 3)}, mean Recall: {round(np.mean(recalls), 3)}')
