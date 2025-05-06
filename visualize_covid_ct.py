import os
import random
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COVID CT Scans dataset samples.")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to CT images directory')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to annotation masks directory')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to visualize')
    parser.add_argument('--overlay', action='store_true', help='Show overlay of mask on image')
    return parser.parse_args()

def get_filenames(images_dir):
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.npy')]

def visualize_sample(image_path, mask_path, overlay=False):
    image_np = np.load(image_path)  # raw CT in Hounsfield units
    mask_np  = np.load(mask_path)   # binary mask

    # pick a 2D slice for display
    if image_np.ndim == 2:
        image_disp = image_np
    elif image_np.ndim == 3 and image_np.shape[-1] == 3:
        # if it really is RGB already, convert to gray for consistency
        image_disp = np.mean(image_np, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {image_np.shape}")

    # Optionally window & normalize your CT
    # e.g. lung window: [-1000, 400] HU → [0,1]
    wmin, wmax = -1000, 400
    image_clipped = np.clip(image_disp, wmin, wmax)
    image_norm    = (image_clipped - wmin) / (wmax - wmin)

    ncols = 3 if overlay else 2
    fig, axs = plt.subplots(1, ncols, figsize=(12, 4))

    # raw CT
    axs[0].imshow(image_norm, cmap='gray')
    axs[0].set_title('Image')
    axs[0].axis('off')

    # mask
    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')

    if overlay:
        # show the windowed CT
        axs[2].imshow(image_norm, cmap='gray')
        # overlay the mask with some transparency
        axs[2].imshow(mask_np, cmap='jet', alpha=0.4)  
        axs[2].set_title('Overlay')
        axs[2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    try:
        filenames = get_filenames(args.images_dir)
    except FileNotFoundError as e:
        print(e)
        return
    if not filenames:
        print("No .npy images found in the specified directory.")
        return
    samples = random.sample(filenames, min(args.num_samples, len(filenames)))
    for fname in samples:
        image_path = os.path.join(args.images_dir, fname + '.npy')
        mask_path = os.path.join(args.masks_dir, fname + '.npy')
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Skipping {fname}: image or mask not found.")
            continue
        visualize_sample(image_path, mask_path, overlay=args.overlay)

if __name__ == "__main__":
    main()

