import torch
from data_setup import create_dataloaders
import model_builder
from lossYoloV1 import YoloV1Loss
import albumentations as A
from pathlib import Path
from albumentations.pytorch import ToTensorV2

# Constants
data_path = Path('../data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S, B, C = 7, 2, 1
EPOCHES = 100
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
transform = A.Compose([
    A.Resize(448, 448),
    A.HorizontalFlip(0.5),
    A.Affine(
        translate_percent=(-0.1, 0.1),
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        border_mode=0,
        p=0.5
    ),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
],bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    clip=True
))

def check_model(train_dataloader, val_dataloader, model, loss_fn):
    model.eval()

    X, Y = next(iter(train_dataloader))

    X = X.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        output = model(X)
        train_eval_loss = loss_fn(output, Y)

    print("Training data in eval mode:", train_eval_loss.item())

    X, Y = next(iter(val_dataloader))

    X = X.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        output = model(X)
        val_loss = loss_fn(output, Y)

    print("Validation loss:", val_loss.item())

def main():
    checkpoint = torch.load(
        "../model/Epoch-1.pth",
        map_location=device,
        weights_only=True
    )
    model = model_builder.YoloV1_model(
        split_size=S,
        num_boxes=B,
        num_classes=C,
        in_channels=3
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_fn = YoloV1Loss(S, B, C)
    train_dataloader, val_dataloader = create_dataloaders(
        train_dir=data_path/'images/train',
        val_dir=data_path/'images/val',
        train_batch_size=4,
        val_batch_size=4,
        train_label_dir=data_path/'labels/train',
        val_label_dir=data_path/'labels/val',
        train_shuffle=True,
        val_shuffle=False,
        is_yolo=True,
        transform=transform
    )

    check_model(train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=model, loss_fn=loss_fn)


if __name__ == "__main__":
    main()