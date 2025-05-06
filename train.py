import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchsummary import summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.unet import UNet
from utils.general import strip_optimizers, random_seed, add_weight_decay
from utils.loss import DiceCELoss, DiceLoss


# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class COVIDCTDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        if file_list is not None:
            # Remove .npy extension if present, so we always add it later
            self.filenames = [f[:-4] if f.endswith('.npy') else f for f in file_list]
        else:
            self.filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.npy')]
        self.transform = transform

        # Filter: keep only files present in both images and masks directories
        images_set = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.npy'))
        masks_set = set(os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.lower().endswith('.npy'))
        both = images_set & masks_set
        orig_len = len(self.filenames)
        self.filenames = [f for f in self.filenames if f in both]
        if len(self.filenames) < orig_len:
            print(f"Filtered {orig_len - len(self.filenames)} entries not present in both images and masks.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image_file = fname + '.npy'
        mask_file = fname + '.npy'
        image = np.load(os.path.join(self.images_dir, image_file)).astype(np.float32)
        mask = np.load(os.path.join(self.masks_dir, mask_file)).astype(np.float32)
        # Windowing and normalization for CT (lung window)
        wmin, wmax = -1000, 400
        image = np.clip(image, wmin, wmax)
        image = (image - wmin) / (wmax - wmin)
        # Add channel dimension if needed
        if image.ndim == 2:
            image = image[None, ...]
        if mask.ndim == 2:
            mask = mask[None, ...]
        # Optionally apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        # Convert to torch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()  # Ensure mask is LongTensor for one_hot
        return image, mask


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="UNet training arguments")

    # Data parameters
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="Directory containing the dataset (default: './data')"
    )
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for input image size (default: 0.5)")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to COVID CT images (.npy)')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to COVID CT masks (.npy)')

    # Model parameters
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes (default: 2)")
    parser.add_argument("--weights", type=str, default="", help="Path to pretrained model weights (default: '')")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: 8)"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-8, help="Weight decay (default: 1e-8)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (default: 0.9)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Frequency of printing training progress (default: 10)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume training from (default: '')"
    )

    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only."
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="weights",
        help="Directory to save model weights (default: 'weights')"
    )

    parser.add_argument('--num_folds', type=int, default=5, help='Number of cross-validation folds')

    args = parser.parse_args()
    return args


def read_split_file(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    batch_loss = []

    for batch_idx, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image = image.to(device)
        target = target.to(device)
        # Squeeze channel dim if present (B, 1, H, W) -> (B, H, W)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_loss.append(loss.item())

        # Print and log progress at specified frequency
        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Train: [{epoch:>3d}][{batch_idx + 1:>4d}/{len(data_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Time: {(time.time() - start_time):.3f}s "
                f"LR: {lr:.7f}"
            )
    logging.info(f"Avg batch loss: {np.mean(batch_loss):.7f}")


@torch.inference_mode()
def evaluate(model, data_loader, device, conf_threshold=0.5):
    model.eval()
    dice_score = 0
    criterion = DiceLoss()

    for image, target in tqdm(data_loader, total=len(data_loader)):
        image, target = image.to(device), target.to(device)
        # Squeeze channel dim if present (B, 1, H, W) -> (B, H, W)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        output = model(image)

        if model.num_classes == 1:
            output = F.sigmoid(output) > conf_threshold

        dice_loss = criterion(output, target)
        dice_score += 1 - dice_loss.item()  # Ensure dice_loss is a scalar

    average_dice_score = dice_score / len(data_loader)

    return average_dice_score, dice_loss.item()


def main(params):
    random_seed()
    os.makedirs(params.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: [{device}]")

    if params.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Cross-validation loop
    for fold in range(params.num_folds):
        logging.info(f"========== Fold {fold+1}/{params.num_folds} ==========")
        # Read split files
        train_split = read_split_file(os.path.join(params.data, f"train_new{fold}.txt"))
        valid_split = read_split_file(os.path.join(params.data, f"valid_new{fold}.txt"))

        # TensorBoard writer per fold with timestamp
        timestamp = time.strftime("%m_%d_%y-%H_%M_%S")
        writer = SummaryWriter(log_dir=os.path.join(params.save_dir, f"runs_fold{fold}_{timestamp}"))

        # Datasets
        train_data = COVIDCTDataset(
            images_dir=params.images_dir,
            masks_dir=params.masks_dir,
            file_list=train_split,
            transform=None
        )
        valid_data = COVIDCTDataset(
            images_dir=params.images_dir,
            masks_dir=params.masks_dir,
            file_list=valid_split,
            transform=None
        )

        # DataLoaders
        train_loader = DataLoader(
            train_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            shuffle=True,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            shuffle=False,
            pin_memory=True
        )

        model = UNet(in_channels=1, num_classes=params.num_classes)
        model.to(device)

        parameters = add_weight_decay(model)
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=params.lr,
            weight_decay=params.weight_decay,
            momentum=params.momentum,
            foreach=True
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
        scaler = torch.cuda.amp.GradScaler() if params.amp else None
        criterion = DiceCELoss()

        start_epoch = 0
        if params.resume:
            checkpoint = torch.load(f"{params.resume}", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            if params.amp:
                scaler.load_state_dict(checkpoint["scaler"])

        logging.info(
            f"Network: [UNet]:\n"
            f"\t{model.in_channels} input channels\n"
            f"\t{model.num_classes} output channels (number of classes)"
        )
        summary(model, (1, 256, 256))

        for epoch in range(start_epoch, params.epochs):
            train_one_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                lr_scheduler,
                device,
                epoch,
                print_freq=params.print_freq,
                scaler=scaler
            )
            # Evaluate on validation set
            dice_score, dice_loss = evaluate(model, valid_loader, device)
            lr_scheduler.step(dice_score)
            # TensorBoard logging
            writer.add_scalar("Loss/valid", dice_loss, epoch)
            writer.add_scalar("Dice/valid", dice_score, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            logging.info(f"Fold {fold} | Epoch {epoch} | Dice Score: {dice_score:.7f} | Dice Loss: {dice_loss:.7f}")
            if params.amp:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, f"{params.save_dir}/checkpoint_fold{fold}.pth")

        writer.close()
        # Strip optimizers & save weights for this fold
        strip_optimizers(f"{params.save_dir}/checkpoint_fold{fold}.pth", save_f=f"{params.save_dir}/last_fold{fold}.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
