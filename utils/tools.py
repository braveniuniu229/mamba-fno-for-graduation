import csv,os
import torch
import logging
import json
def write_to_csv(file_path, epoch, loss):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(epoch, model, optimizer, loss, is_best=False, checkpoint_save_path=None):
    # 确保目录存在
    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,

    }
    if is_best:
        filename = os.path.join(checkpoint_save_path, "checkpoint_best.pth")
        torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return start_epoch, best_loss
    else:
        logging.info(f"No checkpoint found at '{checkpoint_path}'")
        return 0, float('inf')
def save_args(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)