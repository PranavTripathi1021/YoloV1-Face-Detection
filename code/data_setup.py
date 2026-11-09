from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import os
import numpy as np
import torch
import albumentations as A

class faceRecoDataset(Dataset):
    def __init__(self, image_dir, label_dir, is_yolo:bool=False, transform=None, S:int=7, C:int = 1):
        """
        image_dir-All images must be inside this folder
        label_dir-All labels must be inside this folder
        transform-Transforms must be of type albumentation.Compose
        """
        if(isinstance(image_dir, Path)):
            self.image_list = list(image_dir.glob('*.jpg'))
        else:
            self.image_list = list(Path(image_dir).glob('*.jpg'))
        if(isinstance(label_dir, Path)):
            self.label_dir = label_dir
        else:
            self.label_dir = Path(label_dir)
        self.transform = transform
        self.is_yolo = is_yolo
        self.S = S
        self.C = C
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        image_path = self.image_list[index]
        image_name = image_path.stem
        label_path = self.label_dir/f"{image_name}.txt"
        img = Image.open(image_path).convert('RGB') # This gurantees that the image will have 3 channels
        img_np_arr = np.array(img, dtype=np.uint8)
        if os.path.exists(label_path) and os.path.getsize(label_path)>0:
            if(self.is_yolo):
                label_np_arr = np.loadtxt(label_path, usecols=(1, 2, 3, 4), dtype=np.float32)
            else:
                label_np_arr = np.loadtxt(label_path, usecols=(2, 3, 4, 5), dtype=np.float32)
            if label_np_arr.ndim == 1:
                label_np_arr = np.expand_dims(label_np_arr, axis=0)
        else:
            label_np_arr = np.empty((0, 4), dtype=np.float32)
        
        if len(label_np_arr) >0:
            bboxes = label_np_arr.tolist()
            class_labels = [0]*len(bboxes)
        else:
            bboxes = []
            class_labels = []
        
        if self.transform:
            transformed = self.transform(image=img_np_arr, bboxes=bboxes, class_labels=class_labels)
            img_tensor = transformed['image']
            transformed_label = transformed['bboxes']
            transformed_classes = transformed['class_labels']
        else:
            img_tensor = torch.from_numpy(img_np_arr).permute(2, 0, 1).float()/255.0
            transformed_label = bboxes
            transformed_classes = class_labels
        
        
        yolo_target = torch.zeros((self.S, self.S, self.C + 5))
        if len(transformed_label) != 0:
            for i in range(len(transformed_label)):
                x, y, w, h = transformed_label[i]
                class_idx = int(transformed_classes[i])

                cellX, cellY = int(x*self.S), int(y*self.S)
                cellX = min(max(cellX, 0), self.S - 1)
                cellY = min(max(cellY, 0), self.S - 1)

                if(yolo_target[cellY, cellX, self.C] == 1.0):
                    continue
                yolo_target[cellY, cellX, class_idx] =1.0
                yolo_target[cellY, cellX, self.C] = 1.0

                yolo_target[cellY, cellX, self.C + 1] = (x * self.S) - cellX
                yolo_target[cellY, cellX, self.C + 2] = (y * self.S) - cellY
                yolo_target[cellY, cellX, self.C + 3] = w
                yolo_target[cellY, cellX, self.C + 4] = h
        
        return img_tensor, yolo_target





def collate_fn(batch):
    images, targets = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.stack(targets, dim=0),
    )


NUM_WORKERS = 4 # Consider changing it if problem arises

def create_dataloaders(
        train_dir:str|Path,
        val_dir:str|Path,
        train_batch_size:int | None,
        val_batch_size:int | None,
        train_label_dir:str|Path|None = None,
        val_label_dir:str|Path|None = None,
        labels2_dir:str|Path|None = None,
        is_yolo:bool = False,
        transform_train:A.Compose|None = None,
        transform_val:A.Compose|None = None,
        num_workers:int = NUM_WORKERS,
        train_shuffle:bool = True,
        val_shuffle:bool = False
):
    if not is_yolo:
        train_data = faceRecoDataset(
            image_dir=train_dir,
            label_dir=labels2_dir,
            is_yolo=is_yolo,
            transform=transform_train
        )
        val_data = faceRecoDataset(
            image_dir=val_dir,
            label_dir=labels2_dir,
            is_yolo=is_yolo,
            transform=transform_val
        )
    else:
        train_data = faceRecoDataset(
            image_dir=train_dir,
            label_dir=train_label_dir,
            is_yolo=is_yolo,
            transform=transform_train
        )
        val_data = faceRecoDataset(
            image_dir=val_dir,
            label_dir=val_label_dir,
            is_yolo=is_yolo,
            transform=transform_val
        )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(num_workers >0),
        prefetch_factor=2 if num_workers>0 else None
    )
    test_loader = DataLoader(
        dataset=val_data,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(num_workers >0),
        prefetch_factor=2 if num_workers>0 else None
    )
    return train_loader, test_loader
