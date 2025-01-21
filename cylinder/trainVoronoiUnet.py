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
from utils.tools import save_checkpoint, count_parameters
import tqdm
from tools.visualization import save_error,save_prediction
import numpy as np


# 保存损失到 CSV



# Training function
def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    l1_loss_func = L1Loss()
    total_l1_loss = 0
    total_max_ae_loss = 0

    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(1)

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
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)



            l1_loss = l1_loss_func(outputs, labels)
            max_ae_loss = max_aeLoss(outputs, labels)

            total_l1_loss += l1_loss.item()
            total_max_ae_loss += max_ae_loss.item()
            for i in range(20):
                truevalues = labels[i].cpu().numpy()
                predict = outputs[i].cpu().numpy()
                error_file_name = os.path.join("voronoiUnet_checkpoints", f'time_step{i}_error.png')
                predicted_file_name = os.path.join("voronoiUnet_checkpoints", f'time_step{i}_predicted.png')
                save_error(abs(truevalues-predict),error_file_name)
                save_prediction(predict,predicted_file_name)
    avg_l1_loss = total_l1_loss / len(test_loader)
    avg_max_ae_loss = total_max_ae_loss / len(test_loader)
    print(f"Test: Avg L1 Loss: {avg_l1_loss:.4f}, Avg Max_AE Loss: {avg_max_ae_loss:.4f}")
    return avg_l1_loss, avg_max_ae_loss





# Save predicted values for five specific coordinates

def val(args):
    device = torch.device(args.device)

    # Initialize the model
    model = voronoiUNet().to(device)

    # Load the checkpoint
    checkpoint_path = os.path.join(args.save_dir, "checkpoint_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Set the model to evaluation mode
    model.eval()

    # DataLoader for testing/validation
    test_dataset = CylinderDatasetVoronoi1D(data_path=args.path, train=False, train_ratio=0.8, random_points=True, num_points=16)
    test_loader = DataLoader(test_dataset, batch_size=51, shuffle=False)

    total_l1_loss = 0
    total_max_ae_loss = 0
    total_samples = 0

    # Evaluate the model
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(test_loader), desc="Validation", leave=True, colour='white')
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze(1).reshape(outputs.shape[0], -1)
            labels = labels.reshape(outputs.shape[0], -1)

            l1_loss = L1Loss()(outputs, labels)
            max_ae_loss = max_aeLoss(outputs, labels)

            total_l1_loss += l1_loss.item()
            total_max_ae_loss += max_ae_loss.item()
            total_samples += inputs.size(0)

            pbar.set_postfix(l1_loss=l1_loss.item(), max_ae_loss=max_ae_loss.item())
            pbar.update(1)

    avg_l1_loss = total_l1_loss / total_samples
    avg_max_ae_loss = total_max_ae_loss / total_samples

    print(f"Validation L1 Loss: {avg_l1_loss:.4f}")
    print(f"Validation MaxAE Loss: {avg_max_ae_loss:.4f}")

    # Optionally, save validation results or metrics here
    return avg_l1_loss, avg_max_ae_loss

def record(model, testloader, top_5_coords, device, file_name="voronoiunet_predicted_values.csv"):
    predicted_values = []

    model.eval()
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)

            # Get model predictions
            predictions = model(inputs).squeeze(1).reshape(51, -1)  # 51 time steps, multiple points

            for coord in top_5_coords:
                i, j = coord
                point_pred_values = predictions[:, i * 199 + j].cpu().numpy()
                predicted_values.append(point_pred_values)

    np.savetxt(file_name, np.array(predicted_values).T, delimiter=",")
    print(f"Predicted values saved to {file_name}")


# Get five predefined coordinates
def get_top_5_coords():
    return [(1, 66), (1, 67), (0, 66), (0, 68), (0, 67)]



def trainUNet(model, train_loader, val_loader, optimizer, scheduler, device, checkpoint_dir, num_epochs=500, save_interval=5):
    model.to(device)
    best_l1_loss = float('inf')  # Initialize best L1 loss to infinity
    best_max_ae_loss = float('inf')  # Initialize best Max AE loss to infinity

    for epoch in range(1, num_epochs + 1):
        # Training one epoch
        train_l1_loss, train_max_ae_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)

        # Every 5 epochs, perform validation and save the model if better
        if epoch % save_interval == 0:
            val_l1_loss, val_max_ae_loss = test_model(model, val_loader, device)

            # Check if the current model has the best L1 loss
            if val_l1_loss < best_l1_loss:
                best_l1_loss = val_l1_loss
                best_max_ae_loss = val_max_ae_loss
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_l1_loss': best_l1_loss,
                    'best_max_ae_loss': best_max_ae_loss,
                }, checkpoint_path)
                print(f"Saved model checkpoint at epoch {epoch} with L1 Loss: {best_l1_loss:.4f}, Max AE Loss: {best_max_ae_loss:.4f}")
            else:
                print(f"Validation did not improve. Best L1 Loss: {best_l1_loss:.4f}, Best Max AE Loss: {best_max_ae_loss:.4f}")

    print(f"Training completed. Best L1 Loss: {best_l1_loss:.4f}, Best Max AE Loss: {best_max_ae_loss:.4f}")
    return best_l1_loss, best_max_ae_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = voronoiUNet().to(device)

    # Load the model checkpoint
    checkpoint_dir = "voronoiUnet_checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # DataLoader for testing
    test_dataset = CylinderDatasetVoronoi1D(data_path='../data/cylinder.npy', train=False)
    test_loader = DataLoader(test_dataset, batch_size=51, shuffle=False)

    # Call the test function with the loaded model
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()