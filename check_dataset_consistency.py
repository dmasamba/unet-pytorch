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

def main():
    parser = argparse.ArgumentParser(description="Check and clean consistency between images and masks in dataset.")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to masks directory')
    parser.add_argument('--clean', action='store_true', help='Remove unmatched images and masks')
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

if __name__ == "__main__":
    main()
