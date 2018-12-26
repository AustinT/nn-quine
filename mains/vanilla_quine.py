"""
This example trains a vanilla neural network quine
"""

from argparse import ArgumentParser
import sys
import os
import json
import itertools
import pathlib
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import nn

# Custom code
sys.path.append(os.getcwd())
from utils import saver, ml_logging, config as cfg, formulas
from models.nnquines import VanillaQuine

# Arguments
parser = ArgumentParser(description="Run machine learning training")
parser.add_argument("config_file", help="Path to json config file")
parser.add_argument("--device", default=None)
parser.add_argument("--load", default=None, const=-1, nargs="?", type=int,
        help="Load checkpoint. No argument to load latest checkpoint, "
        "or a number to load checkpoint from a particular epoch")
parser.add_argument("--no-train", action="store_true")
parser.add_argument("--eval", action="store_true")

# Functions to make models
def get_models(configs, device):
    nnq = VanillaQuine(n_hidden=configs["n_hidden"], n_layers=configs["n_layers"],
            act_func=getattr(nn, configs["act_func"])).to(device)
    return nnq

# Main program
# Putting training in a main block ensures that model-building functions can be called from elsewhere
if __name__ == "__main__":

    args = parser.parse_args()

    # Load configs
    with open(os.path.join(args.config_file)) as f:
        configs = json.load(f)
    expt_name = cfg.get_expt_name(args.config_file, configs)
    log_dir = ml_logging.get_log_dir(expt_name)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(cfg.expt_summary(expt_name, configs))

    # Setup device
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # Make model
    # Normally would come from model file, but here we are just using a simple file
    # Remember to move models to the correct device!
    nnq = get_models(configs, device)
    model_list = [nnq]

    # Make the data for this
    data = torch.eye(nnq.num_params, device=device)
    index_list = list(range(nnq.num_params))

    # Setup optimizer(s) and loss function(s)
    optimizer = getattr(torch.optim, configs.get("optimizer", "Adam"))
    optimizer = optimizer(itertools.chain(*[model.parameters() for model in model_list]), lr=configs["learning_rate"])
    
    # Learning rate scheduler
    lr_scheduler = configs.get("lr_scheduler", None)
    if lr_scheduler is not None:
        cls_name = lr_scheduler.pop("class_name")
        lr_scheduler = getattr(torch.optim.lr_scheduler, cls_name)(optimizer, **lr_scheduler)


    # Set up logger
    logger = ml_logging.Logger(log_dir)

    # Load model
    if args.load is None:
        start_epoch = 0
    else:
        start_epoch = saver.load_checkpoint(model_list, log_dir, epoch=args.load,
                optimizer=optimizer, lr_scheduler=lr_scheduler)

    # Do training
    if args.no_train:
        print("SKIPPING TRAINING")
    else:
        print("Starting training...")
        for epoch in trange(start_epoch, configs["num_epochs"], desc="Epoch"):
            for model in model_list:
                model.train()

            # Shuffle the list
            random.shuffle(index_list)
            
            # Track losses for each batch
            total_loss = 0.
            avg_relative_error = 0.

            # Go through params one at a time (difficult to put into batch)
            loss = 0.
            optimizer.zero_grad()
            for pos, param_idx in enumerate(tqdm(index_list, leave=False)):
                idx_vector = data[param_idx]
                param = nnq.get_param(param_idx)
                pred_param = nnq(idx_vector)
                mse = (param - pred_param)**2
                loss = loss + mse
                total_loss += mse.item()
                avg_relative_error += formulas.relative_difference(pred_param.item(), param.item())

                if ((pos+1) % configs["batch_size"]) == 0 or pos+1==nnq.num_params:
                    loss.backward()
                    optimizer.step()
                    loss = 0.
                    optimizer.zero_grad()

            # Write to log files
            logger.scalar_summary('mse_loss', total_loss / nnq.num_params, epoch)
            logger.scalar_summary('rel_error', avg_relative_error / nnq.num_params, epoch)

            # Make an overall histogram of the weights
            all_weights = [p.detach().cpu().numpy().flatten() for p in nnq.param_list]
            for w, name in zip(all_weights, nnq.param_names):
                logger.histo_summary(name, w, epoch, bins=20)
            all_weights = np.concatenate(all_weights)
            logger.histo_summary("params", all_weights, epoch, bins=50)

            # Saving and testing
            if epoch % configs.get('save_freq', int(1e6)) == 0:
                saver.save_checkpoint(model_list, log_dir, epoch,
                        optimizer=optimizer, lr_scheduler=lr_scheduler)
            
            # PUT ANY TESTING HERE (the kind that happens every epoch)
            for model in model_list:
                model.eval()
        
        
        # Save a final checkpoint
        saver.save_checkpoint(model_list, log_dir, configs["num_epochs"],
                optimizer=optimizer, lr_scheduler=lr_scheduler)
                
    if args.eval:
        print("Evaluating...")
        for model in model_list:
            model.eval()
        
        # For this example, evaluate 1 whole epoch
        eval_losses = []
        with torch.no_grad():

            # Shuffle the list
            random.shuffle(index_list)
            
            # Track losses for each batch
            total_loss = 0.

            # Go through params one at a time (difficult to put into batch)
            optimizer.zero_grad()
            for pos, param_idx in enumerate(index_list[0:10]):
                idx_vector = data[param_idx]
                param = nnq.get_param(param_idx).item()
                pred_param = nnq(idx_vector).item()
                mse = (param - pred_param)**2
                print("Param #{}: True= {:.3e} Pred= {:.3e} MSE= {:.3e} REL_ERR= {:.3e}".format(pos, 
                    param, pred_param, mse, formulas.relative_difference(param, pred_param)))
                
    # Close the logger
    logger.close()
    print("\n\nSUCCESSFUL END OF SCRIPT")


