import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint,count_parameters, write_to_csv,save_args
from models.mambawithPOD import MambaPOD_time_FNO
from dataset.cylinderLong import CylinderflowDatasetLSTMBeta,SameLengthBatchSampler
from parsercylinder import parse_args
from tools.visualization import save_error,save_prediction
from tools.loss import max_aeLoss
import numpy as np


# Configure the arguments
""" 
configure extraaaaaa
braveniuniu gogogo


"""
args = parse_args()
args.arch = "MambaPOD_time_FNO"
args.d_model = args.num_points
args.d_model_out = 112*192
args.expand = 2



print(args)
best_loss = float("inf")
file_inside = f"random_{args.random}_numpoints_{args.num_points}"
file_root = os.path.join(f"./experiment_log/{args.arch}_blocks_{args.num_blocks}_dstate_{args.d_state}_modesinner_{args.modes}_modesout_f{args.modes1}_width_f{args.width}", file_inside)
ckpt_dir = os.path.join(file_root, args.ckpt_pth)
fig_dir = os.path.join(file_root, args.fig_pth)
result_dir = os.path.join(file_root, args.result_pth)


# 使用 os.makedirs 递归创建目录
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
#保存训练参数到ckpt
save_args(args,os.path.join(ckpt_dir,"args.json"))
device = torch.device("cuda")


train_dataset = CylinderflowDatasetLSTMBeta(data_path=args.data_pth , train=True, slice_lengths=[50])
train_sampler = SameLengthBatchSampler(train_dataset.slices, batch_size=args.batch_size)
testdataset = CylinderflowDatasetLSTMBeta(data_path=args.data_pth, train=False)
trainloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=None)
testloader = DataLoader(testdataset, batch_size=1, shuffle=False)
# Function to save five points' predicted values to CSV file
def recordpoint(model, testloader, top_5_coords, device, file_name="TPSSM-FNO_predicted_values.csv"):
    predicted_values = []

    model.eval()
    with torch.no_grad():
        for inputs, outputs in testloader:
            inputs = inputs.to(device)

            # Get predictions
            predictions = model(inputs).squeeze(0)

            # Extract the predicted values for the 5 points
            for coord in top_5_coords:
                i, j = coord
                point_pred_values = predictions[:, i * 199 + j].cpu().numpy()  # Adjust the indexing based on model output size
                predicted_values.append(point_pred_values)

    # Save the predicted values to CSV file
    np.savetxt(file_name, np.array(predicted_values).T, delimiter=",")
    print(f"Predicted values saved to {file_name}")

def train():
    global best_loss
    args.best_record = {'epoch': -1, 'valloss': 1e10, 'trainloss': 1e10}
    checkpoint_path = os.path.join(ckpt_dir,'checkpoint_best.pth')


    net = MambaPOD_time_FNO(
        modes1=args.modes1,
        modes2 = args.modes2,
        width = args.width,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        d_state=args.d_state,
        d_model_out=args.d_model_out,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="V2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True
    ).to(device)
    start_epoch = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
    print("total parameters:",count_parameters(net))
    for epoch in range(start_epoch,args.epochs):
        # Training procedure
        net.train()
        train_loss, train_num = 0., 0.
        pbar = tqdm.tqdm(total=len(trainloader), desc=f"Training Epoch {epoch}", leave=True, colour='white')
        for inputs, outputs in trainloader:
            inputs, outputs = inputs.cuda(non_blocking=True), outputs.cuda(non_blocking=True)
            pre = net(inputs)
            loss = F.l1_loss(outputs, pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.shape[0]
            train_num += inputs.shape[0]
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
        train_loss = train_loss / train_num
        write_to_csv(f'{result_dir}/train_log.csv', epoch, train_loss)
        logging.info("Epoch: {}, Avg_loss: {}".format(epoch, train_loss))
        scheduler.step()

        # Validation procedure
        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num = 0., 0.
            with torch.no_grad():
                pbar = tqdm.tqdm(total=len(testloader), desc=f"Validation Epoch {epoch}", leave=True, colour='white')
                for inputs, outputs in testloader:
                    inputs, outputs = inputs.to(device), outputs.to(device)

                    pre = net(inputs)
                    loss = F.l1_loss(outputs, pre)

                    val_loss += loss.item() * inputs.shape[0]
                    val_num += inputs.shape[0]
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

                val_loss = val_loss / val_num
                logging.info("Epoch: {}, Val_loss: {}".format(epoch, val_loss))
                write_to_csv(f'{result_dir}/val_log.csv', epoch, val_loss)
                if val_loss < best_loss:
                    is_best = True
                    best_loss = val_loss
                    save_checkpoint(epoch, net, optimizer, val_loss, is_best, ckpt_dir)
                    print("New checkpoint saved in {}".format(ckpt_dir))
            net.train()


def val():
    # Initialize the model
    net = MambaPOD_time_FNO(
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        d_state=args.d_state,
        d_model_out=args.d_model_out,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="V2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True
    ).to(device)

    # Load the checkpoint
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_loss = 0.0
    total_maxae_loss = 0.0
    total_samples = 0
    saved_samples = 0  # Counter for saved samples

    # Create figure directory
    os.makedirs(fig_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Testing", leave=True, colour='white')
        for batch_idx, (inputs, outputs) in enumerate(testloader):
            # Move data to device
            inputs, outputs = inputs.to(device), outputs.to(device)

            # Forward pass
            pred = net(inputs).squeeze(0)  # Remove batch dim
            outputs = outputs.squeeze(0)

            # Calculate losses
            l1_loss = F.l1_loss(pred, outputs)
            maxae_loss = max_aeLoss(pred, outputs)

            # Accumulate metrics
            total_l1_loss += l1_loss.item() * inputs.size(0)
            total_maxae_loss += maxae_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Save first 50 samples
            if saved_samples < 50:
                # Convert to numpy arrays
                pred_np = pred.cpu().numpy().reshape(-1, 112, 192)  # (2000, 112, 192)
                true_np = outputs.cpu().numpy().reshape(-1, 112, 192)

                # Save first 5 timesteps for each sample
                for t in range(50):
                    # Create unique filenames
                    base_name = f"sample{saved_samples}_t{t}"
                    save_prediction(true_np[t], os.path.join(fig_dir, f"{base_name}_true.png"))
                    save_prediction(pred_np[t], os.path.join(fig_dir, f"{base_name}_pred.png"))
                    save_error(np.abs(true_np[t] - pred_np[t]), os.path.join(fig_dir, f"{base_name}_error.png"))

                saved_samples += 1

            pbar.update(1)
            pbar.set_postfix(l1_loss=l1_loss.item(), maxae_loss=maxae_loss.item())

    # Calculate final metrics
    avg_l1 = total_l1_loss / total_samples
    avg_maxae = total_maxae_loss / total_samples

    print(f"\nTest Results:")
    print(f"MAE: {avg_l1:.6f}")
    print(f"Max-AE: {avg_maxae:.6f}")
    print(f"Saved visualizations for {saved_samples} samples to {fig_dir}")

    return avg_l1, avg_maxae

if __name__ == '__main__':
    train()
    # print("best val loss{}".format(best_loss))
    val()
