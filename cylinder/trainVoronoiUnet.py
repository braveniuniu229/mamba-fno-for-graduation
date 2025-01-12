import argparse
import os
import csv
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import L1Loss
from models.VoronoiUnet import voronoiUNet
from dataset.cylinderdatasetVISION import CylinderDatasetVoronoi1D
from tools.loss import max_aeLoss
from utils.tools import save_checkpoint,count_parameters

# 保存损失到 CSV
def save_losses_to_csv(file_path, losses):
    """
    Save losses to a CSV file.

    Parameters:
    - file_path: Path to save the CSV file
    - losses: A list of dictionaries with epoch and loss values
    """
    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'train_loss', 'test_loss_l1', 'test_loss_max_ae'])
        writer.writeheader()
        writer.writerows(losses)

# Training function
def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    l1_loss_func = L1Loss()
    total_l1_loss = 0
    total_max_ae_loss = 0

    for data in train_loader:
        inputs, labels = data  # Assuming dataset returns (inputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(1)  # Remove the second dimension (batch, 1, h, w) -> (batch, h, w)

        l1_loss = l1_loss_func(outputs, labels)
        max_ae_loss = max_aeLoss(outputs, labels)


        l1_loss.backward()
        optimizer.step()

        total_l1_loss += l1_loss.item()
        total_max_ae_loss += max_ae_loss.item()

    scheduler.step()
    avg_l1_loss = total_l1_loss / len(train_loader)
    avg_max_ae_loss = total_max_ae_loss / len(train_loader)
    print(f"Epoch {epoch}: Avg L1 Loss: {avg_l1_loss:.4f}, Avg Max_AE Loss: {avg_max_ae_loss:.4f}")
    return avg_l1_loss, avg_max_ae_loss


# Test function
def test_model(model, test_loader, device):
    model.eval()
    l1_loss_func = L1Loss()
    total_l1_loss = 0
    total_max_ae_loss = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data  # Assuming dataset returns (inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # Remove the second dimension (batch, 1, h, w) -> (batch, h, w)

            l1_loss = l1_loss_func(outputs, labels)
            max_ae_loss = max_aeLoss(outputs, labels)

            total_l1_loss += l1_loss.item()
            total_max_ae_loss += max_ae_loss.item()

    avg_l1_loss = total_l1_loss / len(test_loader)
    avg_max_ae_loss = total_max_ae_loss / len(test_loader)
    print(f"Test: Avg L1 Loss: {avg_l1_loss:.4f}, Avg Max_AE Loss: {avg_max_ae_loss:.4f}")
    return avg_l1_loss, avg_max_ae_loss


# Main function
def main():
    parser = argparse.ArgumentParser(description="Training VoronoiUNet")
    parser.add_argument("--epochs", type=int, default=300, help="Number of total epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--path", type=str, default='../data/cylinder.npy', help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default="voronoiUnet_checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    # Set up device
    device = torch.device(args.device)

    # Load model
    model = voronoiUNet().to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters.")

    # Load dataset
    train_dataset = CylinderDatasetVoronoi1D(args.path, train=True, train_ratio=0.8, random_points=False, num_points=16)
    test_dataset = CylinderDatasetVoronoi1D(args.path, train=False, train_ratio=0.8, random_points=True, num_points=16)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set up optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    # Load checkpoint if specified
    start_epoch = 1
    best_test_loss = float("inf")
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_test_loss = checkpoint["loss"]
        print(f"Resumed training from epoch {start_epoch}, best test loss so far: {best_test_loss:.4f}")

    # Losses history
    losses_history = []

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}")

        # Training
        train_loss_l1, train_loss_max_ae = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        train_loss = train_loss_l1 + train_loss_max_ae

        # Testing every 5 epochs
        if epoch % 5 == 0:
            test_loss_l1, test_loss_max_ae = test_model(model, test_loader, device)
            test_loss = test_loss_l1 + test_loss_max_ae

            # Save checkpoint if test loss improves
            is_best = test_loss < best_test_loss
            if is_best:
                best_test_loss = test_loss

            save_checkpoint(
                epoch, model, optimizer, test_loss, checkpoint_save_path=args.save_dir, is_best=is_best
            )

            # Save losses
            losses_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss_l1": test_loss_l1,
                "test_loss_max_ae": test_loss_max_ae
            })

    # Save losses to CSV
    save_losses_to_csv(os.path.join(args.save_dir, "losses.csv"), losses_history)
    print("Training complete. Losses saved to 'losses.csv'.")


if __name__ == "__main__":
    main()