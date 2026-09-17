import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class RoadDataset(Dataset):
    def __init__(self, data_dir="./GF_LowGradeRoadDataset/train", image_size=(512, 512), is_train=True, start_sample=0, max_samples=None, shuffle=False):
        # Auto-resolve dataset directory if default not found
        if not os.path.exists(data_dir):
            if os.path.exists("./GF_LowGradeRoadDataset/train"):
                data_dir = "./GF_LowGradeRoadDataset/train"
            elif os.path.exists("./dataset"):
                data_dir = "./dataset"

        self.data_dir = data_dir
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
            
        self.is_train = is_train
        self.samples = []

        # Case 1: Check if data_dir has separate images/ and masks/ subfolders
        images_subfolder = os.path.join(data_dir, "images")
        masks_subfolder = os.path.join(data_dir, "masks")

        if os.path.exists(images_subfolder) and os.path.exists(masks_subfolder):
            extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
            img_paths = []
            for ext in extensions:
                img_paths.extend(glob.glob(os.path.join(images_subfolder, ext)))
            img_paths.sort()

            for img_p in img_paths:
                fname = os.path.basename(img_p)
                mask_p = os.path.join(masks_subfolder, fname)
                if not os.path.exists(mask_p):
                    base_name = os.path.splitext(fname)[0]
                    mask_p = os.path.join(masks_subfolder, f"{base_name}.png")
                self.samples.append((img_p, mask_p))

        # Case 2: Satellite imagery style (*_sat.png / *_sat.jpg and *_mask.png in same folder or subfolders)
        else:
            sat_paths = sorted(list(set(glob.glob(os.path.join(data_dir, "*_sat.*")) + glob.glob(os.path.join(data_dir, "**", "*_sat.*"), recursive=True))))
            if len(sat_paths) > 0:
                for sat_p in sat_paths:
                    ext = sat_p.rsplit(".", 1)[-1]
                    base_prefix = sat_p.rsplit("_sat.", 1)[0]
                    mask_p = f"{base_prefix}_mask.{ext}"
                    if not os.path.exists(mask_p):
                        mask_p = f"{base_prefix}_mask.png"
                    if os.path.exists(mask_p):
                        self.samples.append((sat_p, mask_p))
            else:
                img_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png")))
                for img_p in img_paths:
                    if "_mask" not in img_p:
                        base = os.path.splitext(img_p)[0]
                        mask_p = f"{base}_mask.png"
                        if os.path.exists(mask_p):
                            self.samples.append((img_p, mask_p))

        if shuffle:
            random.seed(42)
            random.shuffle(self.samples)

        # Slice subset using start_sample and max_samples
        start = max(0, start_sample)
        if max_samples is not None and max_samples > 0:
            end = start + max_samples
            self.samples = self.samples[start:end]
        elif start > 0:
            self.samples = self.samples[start:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)

        # Resize to specified training resolution
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=transforms.InterpolationMode.NEAREST)

        # Advanced Data Augmentations for Satellite & Road Segmentation
        if self.is_train:
            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 90, 180, 270 degree Rotations
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Photometric / Color Augmentations for varying lighting & soil contrast
            if random.random() > 0.4:
                brightness_factor = random.uniform(0.85, 1.15)
                image = TF.adjust_brightness(image, brightness_factor)

            if random.random() > 0.4:
                contrast_factor = random.uniform(0.85, 1.25)
                image = TF.adjust_contrast(image, contrast_factor)

        # Convert to Tensors
        image_tensor = TF.to_tensor(image) # (3, H, W)
        mask_tensor = TF.to_tensor(mask)   # (1, H, W)
        
        # Binarize mask tensor (0.0 or 1.0)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

