import os
# Helps to increase batch size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Used to get higher memory in GPU
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
from data_setup import create_dataloaders
import model_builder
from lossYoloV1 import YoloV1Loss
from tqdm import tqdm
import albumentations as A
from pathlib import Path
from albumentations.pytorch import ToTensorV2
import re


# Constants
data_path = Path('../data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S, B, C = 7, 2, 1
EPOCHES = 90
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
VELOCITY = 0.999
transform = A.Compose([
    A.Resize(448, 448),
    A.HorizontalFlip(0.5),
    A.Affine(
        translate_percent=(-0.1, 0.1),
        scale=(1.0, 1.3),
        rotate=(-10, 10),
        border_mode=0,
        p=0.5
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=(3, 5)),
    ], p=0.2),
    A.GaussNoise(p=0.2),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.4),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()
],bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    clip=True
))
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == 'cuda'))


def save_checkpoint(model, optimizer, epoch, loss, scheduler, scaler, path="../model/checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")


def train_fn(train_loader, model, loss_fn, optimizer, scaler):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = torch.zeros((), device=device)
    total_samples = 0
    total_individual_losses = torch.zeros(4, device=device)
    for i, (X, Y) in enumerate(loop):
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):
            output = model(X)
            raw_loss, individual_losses = loss_fn(output, Y)
        loss = raw_loss / X.size(0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) 
        scaler.step(optimizer)
        scaler.update()
        total_loss += raw_loss.detach().float()
        total_individual_losses += individual_losses.detach().float()
        total_samples += X.size(0)
        del loss, output
    epoch_loss = total_loss/total_samples
    total_individual_losses = total_individual_losses/total_samples
    print(f"\nMean loss was {epoch_loss.item():.5f}\nLast individual losses are - {total_individual_losses.tolist()}\n")

def val_fn(val_loader, model, loss_fn, scheduler):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_individual_loss = torch.zeros((4), device=device)
    total_samples = 0
    with torch.no_grad():
        for (X, Y) in val_loader:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == 'cuda')
            ):
                output = model(X)
                loss, individual_loss = loss_fn(output, Y)
            total_loss += loss.detach().float()
            total_individual_loss += individual_loss.detach().float()
            total_samples += X.size(0)
        epoch_loss = total_loss/total_samples
        total_individual_loss = total_individual_loss/total_samples
        scheduler.step(epoch_loss.item())
        print(f"\nThe mean loss in Validataion is : {epoch_loss.item():.6f}\n Individual losses in Validation are {total_individual_loss.tolist()}")
        return epoch_loss.item()


def main():
    model = model_builder.YoloV1_model(
        split_size=S,
        num_boxes=B,
        num_classes=C,
        in_channels=3
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(MOMENTUM, VELOCITY))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)
    model_path = Path('../model/')
    epochs = []
    for filename in os.listdir(model_path):
        match = re.fullmatch(r"Epoch-(\d+)\.pth", filename)
        if match:
            epochs.append(int(match.group(1)))
    if epochs:
        last_epoch = max(epochs)
        checkpoint_path = model_path/f"Epoch-{last_epoch}.pth"
        checkpoint =torch.load(checkpoint_path, map_location=device)
        if('model_state_dict' in checkpoint):
            model.load_state_dict(checkpoint['model_state_dict'])
        if('optimizer_state_dict' in checkpoint):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if('scheduler_state_dict' in checkpoint):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if('scaler_state_dict' in checkpoint):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        START_EPOCH = last_epoch
        print(f"Resuming from epoch {last_epoch} | The previous loss was {checkpoint['loss']}")
    else:
        START_EPOCH = 0
        print("No checkpoint found. Starting from scratch.")

    loss_fn = YoloV1Loss(S, B, C)
    train_dataloader, val_dataloader = create_dataloaders(
        train_dir=data_path/'images/train',
        val_dir=data_path/'images/val',
        train_batch_size=16,
        val_batch_size=16,
        train_label_dir=data_path/'labels/train',
        val_label_dir=data_path/'labels/val',
        train_shuffle=True,
        val_shuffle=False,
        is_yolo=True,
        transform=transform
    )
    for epoch in range(START_EPOCH, EPOCHES):
        print(f"Epoch {epoch + 1}\n--------------------------------------\n")
        train_fn(train_dataloader, model, loss_fn, optimizer, scaler)
        current_loss = val_fn(val_dataloader, model, loss_fn, scheduler)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=(epoch+1),
            loss=current_loss,
            scheduler=scheduler,
            scaler=scaler,
            path=f"../model/Epoch-{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()