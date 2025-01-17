import torch
import os
import numpy as np
import csv
import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.mlp import shallow_decoder
from dataset.cylinderdataset import CylinderDatasetMLP
from models.gappypod import GappyPodWeight1D
from tools.loss import max_aeLoss
from argparse import ArgumentParser

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

args = parse_args()
test_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=False)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
device = torch.device("cuda")

# Load GappyPod
path = '../data/cylinder.npy'
origin_data = np.load(path)[0:100,]
gappy_pod = GappyPodWeight1D(data=origin_data, map_size=384*199, n_components=50, observe_weight=50)
def record(model, testloader, top_5_coords, device, file_name="gappymlp_predicted_values.csv"):
    predicted_values = []

    model.eval()
    val_mae, val_maxae, valnum = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, outputs in testloader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            # Get model predictions
            predictions = model(inputs)
            pres = gappy_pod.reconstruct(predictions, inputs, weight=torch.ones_like(predictions))

            # Ensure the pres tensor is on the same device as outputs before computing loss
            pres = pres.to(device)
            val_mae += F.l1_loss(pres, outputs).item() * inputs.shape[0]
            val_maxae += max_aeLoss(pres, outputs).item() * inputs.shape[0]
            valnum += inputs.shape[0]

            # Extract the prediction values for the 5 points
            for coord in top_5_coords:
                i, j = coord
                # We compute the index from the flattened 2D space
                point_pred_values = pres[:, i * 199 + j].cpu().numpy()  # Adjust indexing based on your model's output size
                predicted_values.append(point_pred_values)

        val_mae = val_mae / valnum
        val_maxae = val_maxae / valnum

    # Save the predicted values to CSV
    np.savetxt(file_name, np.array(predicted_values).T, delimiter=",")
    print(f"Predicted values saved to {file_name}")
    print(f"mae:{val_mae}, maxae:{val_maxae}")
# Select the 5 points coordinates
def get_top_5_coords():
    # Return the fixed 5 coordinates
    return [(1, 66), (1, 67), (0, 66), (0, 68), (0, 67)]

# Test function to load model and test
def test(testloader,device):
    model = shallow_decoder(outputlayer_size=args.output_size, n_sensors=args.n_sensors).to(device)
    checkpoint_path = os.path.join("shallowdecoder", args.ckpt_pth, "checkpoint.pth")

    # Load the model checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Get the 5 points coordinates
    # Get the 5 points coordinates
    top_5_coords = get_top_5_coords()

    # Save the predicted values (pres) using the record function
    record(model, testloader, top_5_coords, device)
def val():
    model = shallow_decoder(outputlayer_size=args.output_size, n_sensors=args.n_sensors).to(device)
    checkpoint_path = os.path.join("shallowdecoder", args.ckpt_pth, "checkpoint.pth")

    # Load the model checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Get the 5 points coordinates (top 5)
    top_5_coords = get_top_5_coords()

    # Initialize validation loss variables
    val_mae, val_maxae, val_num = 0.0, 0.0, 0

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Validation", leave=True, colour='white')
        for inputs, outputs in testloader:
            inputs = inputs.to(device)  # Ensure inputs are on the correct device (GPU or CPU)
            outputs = outputs.to(device)  # Ensure outputs are on the correct device (GPU or CPU)

            # Get model predictions
            predictions = model(inputs)
            pres = gappy_pod.reconstruct(predictions, inputs, weight=torch.ones_like(predictions))

            # Ensure the pres tensor is on the same device as outputs before computing loss
            pres = pres.to(device)

            # Compute MAE and MaxAE
            val_mae += F.l1_loss(pres, outputs).item() * inputs.shape[0]
            val_maxae += max_aeLoss(pres, outputs).item() * inputs.shape[0]
            val_num += inputs.shape[0]

            pbar.update(1)

        # Calculate average MAE and MaxAE
        val_mae /= val_num
        val_maxae /= val_num

    # Print final validation results
    print(f"Validation MAE: {val_mae}")
    print(f"Validation MaxAE: {val_maxae}")

if __name__ == "__main__":
    test(testloader,device)
