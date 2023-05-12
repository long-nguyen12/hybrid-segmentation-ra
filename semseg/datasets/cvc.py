import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io, transforms
from pathlib import Path
from typing import Tuple
import torchvision
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

PALETTE = [
        [0, 0, 0],
        [255, 255, 255]
    ]

class CVC_ClinicDB(Dataset):
    CLASSES = [
        'background',
        'polyp'
    ]

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = Path(root) / 'images' / split 
        self.files = list(img_path.glob('*.png'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    @staticmethod
    def convert2mask(mask):
        h, w = mask.shape[:2]
        seg_mask = np.zeros((h,w, len(PALETTE)))

        for i, label in enumerate(PALETTE):
            seg_mask[:, :, i] = np.all(mask == label, axis=-1)

        return seg_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'annotations').replace('.png', '.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(lbl_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert2mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]
            mask = transformed["mask"]
            
            return image.float(), mask.argmax(dim=2).long()
        
        else:
            return image.float(), mask.long()

def create_dataloaders(dir, split, image_size, batch_size, num_workers=os.cpu_count()):
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    dataset = CVC_ClinicDB(root=dir, split=split, transform=transform)
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