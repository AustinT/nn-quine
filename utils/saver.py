import os
import re
import torch

# Formatting strings (constant)
save_format_str = "checkpoint{:08d}.pth"
save_re_string = r"checkpoint(\d{8}).pth"
assert re.match(save_re_string, save_format_str.format(0)) is not None

def save_checkpoint(model_list, save_dir, epoch, optimizer=None, lr_scheduler=None):
    
    checkpoint = {
            'model_states': [model.state_dict() for model in model_list],
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch
            }
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
    
    torch.save(checkpoint, os.path.join(save_dir, save_format_str.format(epoch)))

def load_checkpoint(model_list, save_dir, epoch=-1, load_to_device_name=None,
        optimizer=None, lr_scheduler=None):
    
    # Search for last checkpoint if no epoch given
    if epoch < 0:
        files = os.listdir(save_dir)
        checkpoint_files = \
                list(filter(lambda s: re.match(save_re_string, s) is not None, files))
        if len(checkpoint_files) == 0:
            print("No save files found to load! Proceding with no loading")
            return 0
        last_file = sorted(checkpoint_files)[-1]
        load_epoch = int(re.match(save_re_string, last_file).group(1))
        full_path = os.path.join(save_dir, last_file)
    else:
        full_path = os.path.join(save_dir, save_format_str.format(epoch))
        load_epoch = epoch

    print("Loading checkpoint from: {}".format(full_path), flush=True)
    checkpoint = torch.load(full_path, map_location=load_to_device_name)
    model_states = checkpoint['model_states']
    assert len(model_states) == len(model_list), (len(model_states), len(model_list))
    for model, state in zip(model_list, model_states):
        model.load_state_dict(state)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return load_epoch + 1

