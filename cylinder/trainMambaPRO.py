import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import save_checkpoint,count_parameters, write_to_csv,save_args
from models.MambaPRO import MambaPRO
from dataset.cylinderdataset import CylinderDatasetLSTM
from parsercylinder import parse_args
# from tools.visualization import plot3x1
from tools.utils import cre
# Configure the arguments
"""
configure extraaaaaa
braveniuniu gogogo


"""
args = parse_args()
args.arch = "MambaPRO"
args.d_model = args.num_points
args.d_model_out = 76416
args.d_state = 1024
args.expand = 2
args.d_conv =3


print(args)
best_loss = float("inf")
file_in = f"random_{args.random}_numpoints_{args.num_points}"
file_mid = os.path.join(f"./{args.arch}_lr_{args.lr}_blocks_{args.num_blocks}_dstate_{args.d_state}", file_in)
ckpt_dir = os.path.join(file_mid, args.ckpt_pth)
fig_dir = os.path.join(file_mid, args.fig_pth)
result_dir = os.path.join(file_mid, args.result_pth)


# 使用 os.makedirs 递归创建目录
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
#保存训练参数到ckpt
save_args(args,os.path.join(ckpt_dir,"args.json"))
device = torch.device("cuda")
dataset_train = CylinderDatasetLSTM(args.data_pth, train=True, train_ratio=0.8, random_points=args.random,
                                   num_points=args.num_points)
dataset_test = CylinderDatasetLSTM(args.data_pth, train=False, train_ratio=0.8, random_points=args.random,
                                  num_points=args.num_points)
trainloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

def train():
    global best_loss
    args.best_record = {'epoch': -1, 'valloss': 1e10, 'trainloss': 1e10}
    net = MambaPRO(
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


    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    print("total parameters:",count_parameters(net))
    for epoch in range(args.epochs):
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
                    with torch.no_grad():
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

def test(index):
    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/cylinder2D/logs/ckpt/mlp_cylinder_8/best_epoch_298_loss_0.00004666.pth'

    # Define data loader
    test_dataset = CylinderDataset(index=index)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)

    # Load trained network
    net = MLP(layers=[8, 128, 1280, 4800, 21504]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, outputs) in enumerate(test_loader):
        N, _ = inputs.shape
        inputs, outputs = inputs.cuda(), outputs.cuda()
        with torch.no_grad():
            pre = net(inputs)
        test_num += N
        test_mae += F.l1_loss(outputs.flatten(1), pre).item() * N
        test_rmse += torch.sum(cre(outputs.flatten(1), pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(outputs.flatten(1) - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(112, 192).cpu().numpy(), './test.png')

    import scipy.io as sio
    sio.savemat('mlp.mat', {
        'true': outputs[-1, 0, :, :].cpu().numpy(),
        'pre': pre[-1, :].reshape(112, 192).cpu().numpy()
    })


if __name__ == '__main__':
    train()
    print("best val loss{}".format(best_loss))
    # test(index=[i for i in range(4900, 4901)])
