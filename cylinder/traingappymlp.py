import torch
import os
import numpy as np
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 引入matplotlib用于绘图
from torch.utils.data import DataLoader
from models.mlp import shallow_decoder
from dataset.cylinderdataset import CylinderDatasetMLP
from models.gappypod import GappyPodWeight1D
from tools.loss import max_aeLoss
from argparse import ArgumentParser
from tools.visualization import save_error,save_prediction

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
    parser.add_argument('--fig_pth', type=str, default="fig", help="Path to save figures")
    parser.add_argument('--log_pth', type=str, default="lstm/logs", help="Path to save logs")
    return parser.parse_args()

args = parse_args()
test_dataset = CylinderDatasetMLP(data_path=args.data_pth, train=False)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
device = torch.device("cuda")

# Load GappyPod
path = '../data/cylinder.npy'
origin_data = np.load(path)[0:100,]
gappy_pod = GappyPodWeight1D(data=origin_data, map_size=384*199, n_components=50, observe_weight=50)

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

    # Initialize validation loss variables
    val_mae, val_maxae, val_num = 0.0, 0.0, 0

    # Set model to evaluation mode
    model.eval()
    fig_pth = os.path.join("gappymlp",args.fig_pth)
    os.makedirs(fig_pth,exist_ok=True)
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




            for i in range(20):
                truevalues = outputs[i].reshape(384, 199).cpu().numpy()
                predict = pres[i].reshape(384, 199).cpu().numpy()
                error_file_name = os.path.join(fig_pth, f'time_step{i}_error.png')
                predicted_file_name = os.path.join(fig_pth, f'time_step{i}_predicted.png')
                save_error(abs(truevalues - predict), error_file_name)
                save_prediction(predict, predicted_file_name)

        # Calculate average MAE and MaxAE
        val_mae /= val_num
        val_maxae /= val_num

    # Print final validation results
    print(f"Validation MAE: {val_mae}")
    print(f"Validation MaxAE: {val_maxae}")

if __name__ == "__main__":
    val()
