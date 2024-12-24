import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint,count_parameters, write_to_csv,save_args
from models.mambawithPOD import MambaPOD_time_FNO
from dataset.cylinderdataset import CylinderDatasetLSTMBeta,SameLengthBatchSampler
from parsercylinder import parse_args
from tools.visualization import plot3x1,generate_gif_from_data
from tools.loss import Max_aeLoss


# Configure the arguments
"""
configure extraaaaaa
braveniuniu gogogo


"""
args = parse_args()
args.arch = "MambaPOD_time_FNO"
args.d_model = args.num_points
args.d_model_out = 76416
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


train_dataset = CylinderDatasetLSTMBeta(data_path=args.data_pth , train=True, slice_lengths=[2, 5, 10, 20,25, 30,40, 51, 80,100])
train_sampler = SameLengthBatchSampler(train_dataset.slices, batch_size=args.batch_size)
testdataset = CylinderDatasetLSTMBeta(data_path=args.data_pth, train=False)
trainloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=None)
testloader = DataLoader(testdataset, batch_size=1, shuffle=False)

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


            # Plotting
            # if epoch % args.plot_freq == 0:
            #     plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(112, 192).cpu().numpy(),
            #             file_name=args.fig_path + f'/epoch{epoch}.png')
            #


def test():
    # 加载模型
    maxaeLoss = Max_aeLoss()

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

    # 加载checkpoint
    checkpoint = torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_loss = 0.0
    total_maxae_loss = 0.0
    total_samples = 0

    fields_list = []
    pres_list = []
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(testloader), desc="Testing", leave=True, colour='white')
        for inputs, outputs in testloader:
            inputs, outputs = inputs.to(device), outputs.to(device)

            pre = net(inputs)
            l1_loss_value = F.l1_loss(pre, outputs).item() * inputs.size(0)
            maxae_loss_value = maxaeLoss(pre, outputs).item() * inputs.size(0)

            total_l1_loss += l1_loss_value
            total_maxae_loss += maxae_loss_value
            total_samples += inputs.size(0)


            # reshape outputs and predictions
            pre_reshaped = pre.view(1, 31, 384, 199)
            outputs_reshaped = outputs.view(1, 31, 384, 199)

            for i in range(0, 31, 5):
                true_values = outputs_reshaped[0, i].cpu().numpy()
                predicted_values = pre_reshaped[0, i].cpu().numpy()

                plot3x1(true_values, predicted_values, file_name=os.path.join(fig_dir, f'figure_{i}.png'))
                # fields_list.append(true_values)
                # pres_list.append(predicted_values)

            pbar.update(1)
        # avg_l1_loss = total_l1_loss / total_samples
        # avg_maxae_loss = total_maxae_loss / total_samples
        # output_gif = os.path.join(fig_dir, 'output.gif')
        # generate_gif_from_data(fields_list, pres_list, output_gif)
        # print(f"GIF saved as {output_gif}")
        #
        # print(f'Average L1 Loss: {avg_l1_loss}, Average Max AE Loss: {avg_maxae_loss}')


if __name__ == '__main__':
    train()
    print("best val loss{}".format(best_loss))
    test()
