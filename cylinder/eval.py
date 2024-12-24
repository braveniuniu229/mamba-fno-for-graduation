import torch
import torch.nn.functional as F
import logging
import os
import tqdm
from torch.utils.data import DataLoader
from utils.tools import load_checkpoint
from models.mambawithPOD import MambaPOD_space
from dataset.cylinderdataset import CylinderDatasetLSTM
from parsercylinder import parse_args
from tools.visualization import plot_results
from tools.utils import cre
import numpy as np
import matplotlib


# Configure the arguments
"""
configure extraaaaaa
braveniuniu gogogo


"""
args = parse_args()
args.arch = "MambaPOD_space"
args.d_model = args.num_points
args.d_model_out = 76416
args.expand = 2



print(args)

file_in = f"random_{args.random}_numpoints_{args.num_points}"
file_mid = os.path.join(f"./experiment_log/{args.arch}_lr_{args.lr}_blocks_{args.num_blocks}_dstate_{args.d_state}", file_in)
ckpt_dir = os.path.join(file_mid, args.ckpt_pth)
fig_dir = os.path.join(file_mid, args.fig_pth)
result_dir = os.path.join(file_mid, args.result_pth)


# 使用 os.makedirs 递归创建目录
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
device = torch.device("cuda")
dataset_test = CylinderDatasetLSTM(args.data_pth, train=False, train_ratio=0.8, random_points=args.random,
                                  num_points=args.num_points)
testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

def eval():
    net = MambaPOD_space(
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
    #load ckpt
    _,loss = load_checkpoint(os.path.join(ckpt_dir,"checkpoint_best.pth"),net,optimizer)
    with torch.no_grad():
        # Training procedure
        net.eval()
        eval_loss, eval_num = 0., 0.
        pbar = tqdm.tqdm(total=len(testloader), desc="eval process", leave=True, colour='white')
        indices = np.linspace(0, args.d_model_out - 1, args.num_points, dtype=int)
        positions =[]
        for i in indices:
            y = i % 199
            x = i // 199
            positions.append([x,y])
        positions = np.array(positions)
        for i,(inputs, outputs) in enumerate(testloader):
            inputs, outputs = inputs.cuda(non_blocking=True), outputs.cuda(non_blocking=True)
            pre = net(inputs)
            loss = F.l1_loss(outputs, pre)


            eval_loss += loss.item() * inputs.shape[0]
            eval_num += inputs.shape[0]
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            precpu = pre.reshape(-1, 384, 199).cpu().numpy()
            gthcpu = outputs.reshape(-1,384,199).cpu().numpy()
            losscpu = precpu-gthcpu
            for j in range(5):
                matplotlib.use('Agg')  # 使用非交互式后端
                gdtpth = os.path.join(fig_dir,f"groundtruth{j}")
                errorpth = os.path.join(fig_dir, f"error{j}")
                prepth = os.path.join(fig_dir, f"pre{j}")
                plot_results(positions,precpu[j],prepth)
                plot_results(positions,losscpu[j],errorpth)
                plot_results(positions,gthcpu[j],gdtpth)
        eval_loss = eval_loss / eval_num
        print(eval_loss)







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
    eval()

    # test(index=[i for i in range(4900, 4901)])
