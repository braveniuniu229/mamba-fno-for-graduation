import torch
import torch.nn.functional as F
import os
import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models.voronoiCNN import VoronoiCNN
from dataset.cylinderdatasetVISION import CylinderDatasetVoronoi1D
from tools.visualization import plot3x1
from tools.loss import max_aeLoss
import numpy as np

# Argument parsing
def parse_args():
    parser = ArgumentParser(description="Training VCNN model")
    parser.add_argument('--data_pth', type=str, default="../data/cylinder.npy")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help="Learning rate decay rate")
    parser.add_argument('--weight_decay_rate', type=float, default=0.8, help="Weight decay rate")
    parser.add_argument('--lr_decay_epoch', type=int, default=100, help="Epoch interval for learning rate decay")
    parser.add_argument('--n_sensors', type=int, default=16, help="Number of input sensors")
    parser.add_argument('--output_size', type=int, default=76416, help="Output size of the model")
    parser.add_argument('--val_interval', type=int, default=5, help="Validation interval")
    parser.add_argument('--ckpt_pth', type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument('--fig_pth', type=str, default="figures", help="Path to save figures")
    parser.add_argument('--log_pth', type=str, default="shallowdecoder/logs", help="Path to save logs")
    return parser.parse_args()

# Exponential learning rate scheduler
def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.9, weight_decay_rate=0.8, lr_decay_epoch=100):
    if epoch % lr_decay_epoch:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_rate
        param_group['weight_decay'] *= weight_decay_rate

# Training function
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    os.makedirs(os.path.join("VoronoiCNN", args.ckpt_pth), exist_ok=True)
    os.makedirs(os.path.join("VoronoiCNN", args.fig_pth), exist_ok=True)
    os.makedirs(args.log_pth, exist_ok=True)

    # Dataset and DataLoader
    train_dataset = CylinderDatasetVoronoi1D(data_path=args.data_pth, train=True)
    test_dataset = CylinderDatasetVoronoi1D(data_path=args.data_pth, train=False)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=51, shuffle=False)

    # Model, optimizer, and scheduler
    model = VoronoiCNN( ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    best_loss = float('inf')
    best_maeloss = float('inf')
    checkpoint_path = os.path.join("VoronoiCNN", args.ckpt_pth, f"checkpoint.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Load checkpoint if available
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")

    # Log files
    train_log_path = os.path.join(args.log_pth, "train_logs.csv")
    val_log_path = os.path.join(args.log_pth, "val_logs.csv")
    with open(train_log_path, "w") as f:
        f.write("epoch,train_loss,train_maxae_loss\n")
    with open(val_log_path, "w") as f:
        f.write("epoch,val_loss,val_maxae_loss\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_maxae_loss, train_num = 0.0, 0.0, 0
        pbar = tqdm.tqdm(total=len(trainloader), desc=f"Training Epoch {epoch}", leave=True, colour='white')

        for inputs, outputs in trainloader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            predictions = model(inputs).squeeze(1)
            loss = F.l1_loss(predictions, outputs)
            maxaeloss = max_aeLoss(predictions, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_maxae_loss += maxaeloss.item() * inputs.size(0)
            train_num += inputs.size(0)
            pbar.set_postfix(loss=loss.item(), maxae_loss=maxaeloss.item())
            pbar.update(1)

        train_loss /= train_num
        train_maxae_loss /= train_num
        print(f"Epoch {epoch}, Training Loss: {train_loss}, Training MaxAE Loss: {train_maxae_loss}")

        # Log training metrics
        with open(train_log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_maxae_loss}\n")

        # Apply custom LR scheduler
        scheduler.step()

        # Validation
        if epoch % args.val_interval == 0:
            model.eval()
            val_loss, val_maxae_loss, val_num = 0.0, 0.0, 0
            with torch.no_grad():
                pbar = tqdm.tqdm(total=len(testloader), desc=f"Validation Epoch {epoch}", leave=True, colour='white')
                for inputs, outputs in testloader:
                    inputs, outputs = inputs.to(device), outputs.to(device)
                    outputs = outputs.reshape(51,-1)
                    predictions = model(inputs).squeeze(1).reshape(51,-1)
                    loss = F.l1_loss(predictions, outputs)
                    maxaeloss = max_aeLoss(predictions, outputs)

                    val_loss += loss.item() * inputs.size(0)
                    val_maxae_loss += maxaeloss.item() * inputs.size(0)
                    val_num += inputs.size(0)
                    pbar.set_postfix(loss=loss.item(), maxae_loss=maxaeloss.item())
                    pbar.update(1)

            val_loss /= val_num
            val_maxae_loss /= val_num
            print(f"Epoch {epoch}, Validation Loss: {val_loss}, Validation MaxAE Loss: {val_maxae_loss}")

            # Log validation metrics
            with open(val_log_path, "a") as f:
                f.write(f"{epoch},{val_loss},{val_maxae_loss}\n")

            # Save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_maeloss = val_maxae_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'max-aeloss': val_maxae_loss
                }, checkpoint_path)
                print(f"New best model saved at {checkpoint_path}")

    print(f"Training completed. Best Validation Loss: {best_loss}, Best Validation MaxAE Loss: {best_maeloss}")


import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from models.voronoiCNN import VoronoiCNN
from dataset.cylinderdatasetVISION import CylinderDatasetVoronoi1D
from tools.loss import max_aeLoss
import tqdm

# Save the predicted values for five specific coordinates
def record(model, testloader, top_5_coords, device, file_name="voronoicnn_predicted_values.csv"):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = VoronoiCNN().to(device)

    # Load the model checkpoint
    checkpoint_path = os.path.join("VoronoiCNN", "checkpoints", "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # DataLoader for testing
    test_dataset = CylinderDatasetVoronoi1D(data_path='../data/cylinder.npy', train=False)
    testloader = DataLoader(test_dataset, batch_size=51, shuffle=False)

    # Get the 5 predefined coordinates
    top_5_coords = get_top_5_coords()

    # Call the record function to save the predicted values to CSV
    record(model, testloader, top_5_coords, device, file_name="voronoicnn_predicted_values.csv")
# Main

def val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = VoronoiCNN().to(device)

    # Load the model checkpoint
    checkpoint_path = os.path.join("VoronoiCNN", args.ckpt_pth, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Set the model to evaluation mode
    model.eval()

    # DataLoader for testing/validation
    test_dataset = CylinderDatasetVoronoi1D(data_path=args.data_pth, train=False)
    testloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    total_loss = 0.0
    total_maxae_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Validation", leave=True, colour='white')
        for inputs, outputs in testloader:
            inputs, outputs = inputs.to(device), outputs.to(device)

            # Get the model predictions
            predictions = model(inputs).squeeze(1).reshape(10, -1)
            outputs = outputs.reshape(10,-1)

            # Calculate L1 loss and MaxAE loss
            loss = F.l1_loss(predictions, outputs)
            maxaeloss = max_aeLoss(predictions, outputs)

            total_loss += loss.item() * inputs.size(0)
            total_maxae_loss += maxaeloss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            pbar.set_postfix(loss=loss.item(), maxae_loss=maxaeloss.item())
            pbar.update(1)

    avg_loss = total_loss / total_samples
    avg_maxae_loss = total_maxae_loss / total_samples

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation MaxAE Loss: {avg_maxae_loss}")

    # Optionally save results or metrics here if necessary (e.g., save to CSV, logging)

if __name__ == "__main__":
    main()
