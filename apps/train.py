import sys, os
sys.path.insert(0, os.getcwd())

import random
import argparse
import numpy as np
import configs
from datetime import timedelta

from lib import trainers

import torch
import torch.distributed as dist
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    # training options
    parser.add_argument('--config', type=str, required=True)

    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='log')
    # parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--eval_freq', type=int, default=0)
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=1000)
    parser.add_argument('--model_keep_interval', type=int, default=5000)

    parser.add_argument('--bs_factor', type=int, default=1, help="batch split factor")
    parser.add_argument("--local_rank", default=-1, type=int)

    # parameter tuning options
    parser.add_argument('--tune', type=str, default='')
    parser.add_argument('--variant', type=int, default=0)

    opt = parser.parse_args()

    assert opt.model_keep_interval % opt.model_save_interval == 0

    return opt



if __name__ == '__main__':

    opt = parse_args()
    opt.local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    global_rank = torch.distributed.get_rank()

    device = torch.device(f"cuda:{opt.local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    if global_rank == 0:
        print(opt)
        os.makedirs(opt.output_dir, exist_ok=True)

    world_size = torch.distributed.get_world_size()

    random.seed(global_rank)
    np.random.seed(global_rank)
    torch.manual_seed(global_rank)

    config = configs.get_config(opt)
    trainer_cls = getattr(trainers, config["trainer"])
    trainer = trainer_cls(global_rank, world_size, device, opt, config)
    trainer.run()

    dist.destroy_process_group()
