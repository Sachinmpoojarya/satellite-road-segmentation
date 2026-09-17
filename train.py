import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet_model import UNet
from dataset_loader import RoadDataset

class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky Loss tailored for satellite road segmentation.
    Addresses severe 95:5 background-to-road class imbalance and forces higher recall on low-grade unpaved roads.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha   # Weight for False Negatives (missing roads)
        self.beta = beta     # Weight for False Positives
        self.gamma = gamma   # Focal focusing exponent
        self.smooth = smooth

    def forward(self, preds, targets):
        eps = 1e-6
        preds_clamped = torch.clamp(preds, eps, 1.0 - eps)

        # Flatten tensors for Tversky metric calculation
        p_flat = preds_clamped.view(-1)
        t_flat = targets.view(-1)

        # True Positives, False Positives, False Negatives
        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky_index), self.gamma)

        # Weighted Focal BCE component
        focal_bce = - (2.0 * targets * torch.log(preds_clamped) + (1.0 - targets) * torch.log(1.0 - preds_clamped)).mean()

        return focal_tversky + 0.5 * focal_bce


def calculate_iou(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).float()
    intersection = (preds_binary * targets).sum().item()
    union = (preds_binary + targets - preds_binary * targets).sum().item()
    if union == 0:
        return 1.0
    return intersection / union

def train_model(data_dir="./GF_LowGradeRoadDataset/train", start_sample=0, max_samples=0, epochs=15, batch_size=8, lr=1e-4, image_size=512, shuffle=True, checkpoint_path="./road_model/best_road_seg_unet.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for training: {device}", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Resolve default dataset path if needed
    if not os.path.exists(data_dir):
        if os.path.exists("./GF_LowGradeRoadDataset/train"):
            data_dir = "./GF_LowGradeRoadDataset/train"
        elif os.path.exists("./dataset"):
            data_dir = "./dataset"

    print(f"Loading dataset from: {os.path.abspath(data_dir)}", flush=True)
    print(f"Dataset resolution: {image_size}x{image_size} | start index = {start_sample}, max count = {max_samples or 'ALL'} | Shuffle: {shuffle}", flush=True)

    train_dataset = RoadDataset(data_dir=data_dir, image_size=(image_size, image_size), is_train=True, start_sample=start_sample, max_samples=max_samples, shuffle=shuffle)
    if len(train_dataset) == 0:
        print(f"No training image pairs found in '{data_dir}'! Check folder structure.", flush=True)
        return

    print(f"Loaded {len(train_dataset)} training image pairs for high-precision training.", flush=True)


    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = True if device == "cuda" else False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Initialize Model with Pretrained ResNet34 Backbone
    model = UNet(in_channels=3, out_channels=1, pretrained=True)

    # If existing weight file exists with same architecture, fine-tune from it
    if os.path.exists(checkpoint_path):
        print(f"Checking existing checkpoint at '{checkpoint_path}'...", flush=True)
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded existing weights for fine-tuning.", flush=True)
        except Exception as e:
            print(f"Starting fresh training with ImageNet pre-trained ResNet34 backbone ({e}).", flush=True)

    model.to(device)

    # Loss, Optimizer, Cosine Learning Rate Scheduler, and AMP Mixed Precision Scaler
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_iou = 0.0

    print(f"\nStarting High-Accuracy ResNet-UNet Training ({image_size}x{image_size})...", flush=True)
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | AMP Mixed Precision: {use_amp} | Initial LR: {lr}\n", flush=True)


    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_iou = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs.float(), masks.float())

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_iou += calculate_iou(outputs, masks) * images.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_iou = running_iou / len(train_dataset)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | IoU Score: {epoch_iou:.4f} | LR: {current_lr:.6f}", flush=True)

        # Save single model checkpoint whenever IoU improves or on final epoch
        if epoch_iou >= best_iou or epoch == epochs:
            best_iou = max(best_iou, epoch_iou)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved best model checkpoint to '{checkpoint_path}' (IoU: {epoch_iou:.4f})", flush=True)

    print("\nTraining completed successfully!", flush=True)
    print(f"Best Model Weights Saved At: {os.path.abspath(checkpoint_path)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-UNet High-Accuracy Road Segmentation Model")
    parser.add_argument("--data_dir", type=str, default="./GF_LowGradeRoadDataset/train", help="Path to training data directory")
    parser.add_argument("--start_sample", type=int, default=0, help="Starting index of dataset images")
    parser.add_argument("--max_samples", type=int, default=0, help="Number of images to train on (0 = all)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=512, help="Input resolution (default: 512)")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle dataset before slicing")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--checkpoint", type=str, default="./road_model/best_road_seg_unet.pth", help="Checkpoint save path")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        start_sample=args.start_sample,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=args.shuffle,
        lr=args.lr,
        checkpoint_path=args.checkpoint
    )

