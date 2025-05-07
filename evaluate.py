import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.unet import UNet
from utils.loss import DiceLoss
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate UNet model on a test set")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to test images (.npy)')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to test masks (.npy)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()
    return args

class TestDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.npy')]
        # Filter to only those with masks
        masks_set = set(os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.lower().endswith('.npy'))
        self.filenames = [f for f in self.filenames if f in masks_set]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = np.load(os.path.join(self.images_dir, fname + '.npy')).astype(np.float32)
        mask = np.load(os.path.join(self.masks_dir, fname + '.npy')).astype(np.int64)
        wmin, wmax = -1000, 400
        image = np.clip(image, wmin, wmax)
        image = (image - wmin) / (wmax - wmin)
        if image.ndim == 2:
            image = image[None, ...]
        if mask.ndim == 2:
            mask = mask[None, ...]
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        return image, mask

def dice_score(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    dice = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            dice.append(np.nan)
        else:
            dice.append((2. * intersection / union).item())
    return dice

def iou_score(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return ious

def precision_recall(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    precisions, recalls = [], []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(in_channels=1, num_classes=args.num_classes)
    state_dict = torch.load(args.model_path, map_location=device)
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].float()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Dataset and loader
    dataset = TestDataset(args.images_dir, args.masks_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    dice_all = []
    iou_all = []
    prec_all = []
    rec_all = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            for i in range(images.size(0)):
                d = dice_score(preds[i].cpu(), masks[i].cpu(), args.num_classes)
                iou = iou_score(preds[i].cpu(), masks[i].cpu(), args.num_classes)
                p, r = precision_recall(preds[i].cpu(), masks[i].cpu(), args.num_classes)
                dice_all.append(d)
                iou_all.append(iou)
                prec_all.append(p)
                rec_all.append(r)

    dice_all = np.array(dice_all)
    iou_all = np.array(iou_all)
    prec_all = np.array(prec_all)
    rec_all = np.array(rec_all)

    print("Evaluation results (per class):")
    for c in range(args.num_classes):
        print(f"Class {c}:")
        print(f"  Dice:      {np.nanmean(dice_all[:, c]):.4f}")
        print(f"  IoU:       {np.nanmean(iou_all[:, c]):.4f}")
        print(f"  Precision: {np.nanmean(prec_all[:, c]):.4f}")
        print(f"  Recall:    {np.nanmean(rec_all[:, c]):.4f}")

    print("\nMean metrics (averaged over classes):")
    print(f"  Dice:      {np.nanmean(dice_all):.4f}")
    print(f"  IoU:       {np.nanmean(iou_all):.4f}")
    print(f"  Precision: {np.nanmean(prec_all):.4f}")
    print(f"  Recall:    {np.nanmean(rec_all):.4f}")

    # --- Plotting and saving metrics ---
    metrics_dir = "eval_metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    classes = [f"Class {c}" for c in range(args.num_classes)]

    # Per-class bar plots
    def plot_bar(metric, values, ylabel, fname):
        plt.figure()
        plt.bar(classes, values)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} per class")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(metrics_dir, fname))
        plt.close()

    plot_bar("Dice", [np.nanmean(dice_all[:, c]) for c in range(args.num_classes)], "Dice", "dice_per_class.png")
    plot_bar("IoU", [np.nanmean(iou_all[:, c]) for c in range(args.num_classes)], "IoU", "iou_per_class.png")
    plot_bar("Precision", [np.nanmean(prec_all[:, c]) for c in range(args.num_classes)], "Precision", "precision_per_class.png")
    plot_bar("Recall", [np.nanmean(rec_all[:, c]) for c in range(args.num_classes)], "Recall", "recall_per_class.png")

    # Boxplots for sample-wise distribution
    def plot_box(metric, values, ylabel, fname):
        plt.figure()
        plt.boxplot([values[:, c][~np.isnan(values[:, c])] for c in range(args.num_classes)], labels=classes)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} distribution per class")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(metrics_dir, fname))
        plt.close()

    plot_box("Dice", dice_all, "Dice", "dice_boxplot.png")
    plot_box("IoU", iou_all, "IoU", "iou_boxplot.png")
    plot_box("Precision", prec_all, "Precision", "precision_boxplot.png")
    plot_box("Recall", rec_all, "Recall", "recall_boxplot.png")

    print(f"\nPlots saved to {metrics_dir}/")

if __name__ == "__main__":
    main()
