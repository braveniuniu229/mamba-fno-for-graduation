import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='cylinder_flow')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--arch', type=str, default='MLP', help='Model architecture')
    parser.add_argument('--weightdecay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--dataset', type=str, default='cylinder', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--tag', type=str, default='baseline', help='Tag for the experiment')
    parser.add_argument('--lr_decay_epoch', type=int, default=100, help='LR decay epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--ckpt_pth',type=str,default="ckpt")
    parser.add_argument('--fig_pth', type=str, default="fig")
    parser.add_argument('--result_pth', type=str, default="result")
    parser.add_argument('--data_pth', type=str, default="../data/flow_cylinder.npy")
    parser.add_argument('--random', type=bool, default=False)
    parser.add_argument('--num_points', type=int, default=16)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--num_blocks',type=int,default=5)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--d_conv', type=int, default=3)
    parser.add_argument('--modes1', type=int, default=90)
    parser.add_argument('--modes2', type=int, default=90)
    parser.add_argument('--width', type=int, default=12)
    parser.add_argument('--modes', type=int, default=16)




    return parser.parse_args()