import torch
import torch.nn.functional as F
import os
import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models.mlp import shallow_decoder
from dataset.cylinderdataset import CylinderDatasetMLP
from tools.visualization import plot3x1
from tools.loss import max_aeLoss

# Argument parsing
def parse_args():
    parser = ArgumentParser(description="Training shallow_decoder model")
    parser.add_argument('--data_pth', type=str, default="../data/cylinder.npy")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=4000, help="Number of training epochs")
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
def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, weight_decay_rate=0.8, lr_decay_epoch=100):
    if epoch % lr_decay_epoch:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_rate
        param_group['weight_decay'] *= weight_decay_rate

# Training function
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    os.makedirs(os.path.join("shallowdecoder", args.ckpt_pth), exist_ok=True)
    os.makedirs(os.path.join("shallowdecoder", args.fig_pth), exist_ok=True)
    os.makedirs(args.log_pth, exist_ok=True)

    # Dataset and DataLoader
    train_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=True)
    test_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=False)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, optimizer, and scheduler
    model = shallow_decoder(outputlayer_size=args.output_size, n_sensors=args.n_sensors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    best_loss = float('inf')
    best_maeloss = float('inf')
    checkpoint_path = os.path.join("shallowdecoder", args.ckpt_pth, f"checkpoint.pth")
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
            predictions = model(inputs)
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
        exp_lr_scheduler(optimizer, epoch, lr_decay_rate=args.lr_decay_rate,
                         weight_decay_rate=args.weight_decay_rate, lr_decay_epoch=args.lr_decay_epoch)

        # Validation
        if epoch % args.val_interval == 0:
            model.eval()
            val_loss, val_maxae_loss, val_num = 0.0, 0.0, 0
            with torch.no_grad():
                pbar = tqdm.tqdm(total=len(testloader), desc=f"Validation Epoch {epoch}", leave=True, colour='white')
                for inputs, outputs in testloader:
                    inputs, outputs = inputs.to(device), outputs.to(device)
                    predictions = model(inputs)
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

# Main
def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
