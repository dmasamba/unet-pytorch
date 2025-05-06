import os
import argparse

def get_basenames(directory):
    return set(os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith('.npy'))

def remove_files(directory, basenames):
    for name in basenames:
        path = os.path.join(directory, name + '.npy')
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed: {path}")

def clean_split_files(split_dir, images_basenames, masks_basenames):
    for fname in os.listdir(split_dir):
        if fname.startswith("train_new") or fname.startswith("valid_new"):
            path = os.path.join(split_dir, fname)
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            # Keep only entries present in both images and masks
            cleaned = [
                line for line in lines
                if ((line[:-4] if line.endswith('.npy') else line) in images_basenames and
                    (line[:-4] if line.endswith('.npy') else line) in masks_basenames)
            ]
            removed = len(lines) - len(cleaned)
            if removed > 0:
                with open(path, "w") as f:
                    for line in cleaned:
                        f.write(line + "\n")
                print(f"Cleaned {removed} missing entries from {fname}")

def main():
    parser = argparse.ArgumentParser(description="Check and clean consistency between images and masks in dataset.")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to masks directory')
    parser.add_argument('--clean', action='store_true', help='Remove unmatched images and masks')
    parser.add_argument('--clean_splits', action='store_true', help='Clean split files by removing missing images')
    parser.add_argument('--split_dir', type=str, default=None, help='Directory containing split files (default: parent of images_dir)')
    args = parser.parse_args()

    images = get_basenames(args.images_dir)
    masks = get_basenames(args.masks_dir)

    images_without_masks = images - masks
    masks_without_images = masks - images

    print(f"Total images: {len(images)}")
    print(f"Total masks: {len(masks)}")
    print(f"Images without masks: {len(images_without_masks)}")
    print(f"Masks without images: {len(masks_without_images)}")

    if images_without_masks:
        print("\nImages without corresponding masks:")
        for name in sorted(images_without_masks):
            print(f"  {name}.npy")
    if masks_without_images:
        print("\nMasks without corresponding images:")
        for name in sorted(masks_without_images):
            print(f"  {name}.npy")

    if args.clean:
        if images_without_masks:
            print("\nRemoving unmatched images...")
            remove_files(args.images_dir, images_without_masks)
        if masks_without_images:
            print("\nRemoving unmatched masks...")
            remove_files(args.masks_dir, masks_without_images)

    if args.clean_splits:
        split_dir = args.split_dir or os.path.dirname(args.images_dir)
        print(f"\nCleaning split files in {split_dir} ...")
        clean_split_files(split_dir, images, masks)

if __name__ == "__main__":
    main()
