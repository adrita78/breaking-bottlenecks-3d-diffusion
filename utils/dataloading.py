import os
import argparse
from featurization import construct_loader
#from ESLapPE import EquivStableLapPENodeEncoder
#from ESLapPE import set_cfg_posenc
#from torch_geometric.graphgym.config import cfg
from torch_geometric.data import Batch
import torch
from torch_geometric.nn import global_add_pool
#from gaussian_diffusion import VP_Diffusion
#from gaussian_diffusion import get_named_beta_schedule
#from gaussian_diffusion import mean_flat
import numpy as np
import torch as th
import enum
#import enum
#from abc import ABC, abstractmethod
#import torch.distributed as dist
#from gaussian_diffusion import UniformSampler
#from yacs.config import CfgNode as CN
#from torch_geometric.graphgym.register import register_config
#set_cfg_posenc(cfg)
# Step 1: Setup cache_dir
cache_dir = "E:/rdkit/cache_7"
os.makedirs(cache_dir, exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#encoder = EquivStableLapPENodeEncoder(dim_emb=74)
#class LossType(enum.Enum):
    #MSE = enum.auto()  # use raw MSE loss

    #PH = enum.auto() # Pseudo-Huber
    #PL = enum.auto() # Pseudo-LPIPS

#num_timesteps = 1000
#betas = get_named_beta_schedule(
    #schedule_name="linear", 
    #num_diffusion_timesteps=num_timesteps,
    #beta_start=1e-4, 
    #beta_end=0.02
#)    

#diffusion = VP_Diffusion(
    #betas=betas,
    #loss_type=LossType.MSE,  # or LossType.PL
    #c=0.0
#)

#sampler = UniformSampler(diffusion)
#device = torch.device("cpu")

# Step 2: Define parser
parser = argparse.ArgumentParser(description="Drug dataset preprocessing")

parser.add_argument("--data_dir", type=str, default="E:/rdkit/rdkit_folder")
parser.add_argument("--split_path", type=str, default="E:/rdkit/split_new_1.npy")
parser.add_argument("--dataset", type=str, default="drugs")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--limit_train_mols", type=int, default=40000)
parser.add_argument("--max_confs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=32)

args = parser.parse_args([]) 
args.cache = f"{cache_dir}/drugs_split_{os.path.basename(args.split_path)}_maxconfs_{args.max_confs}.cache"

train_loader, val_loader = construct_loader(args, modes=("train", "val"))
for batch_idx, batch in enumerate(train_loader):
    #print(batch)
    pass

x= batch.x
print("shape of x:", x.shape)
#device = "cuda" if torch.cuda.is_available() else "cpu"
#context = torch.randn_like(batch.x, dtype=torch.float).to(device)
#context_1 = context[batch.batch]  
#num_graphs= (batch.ptr.shape[0]-1)
#t = torch.randint(0, 1000, (num_graphs,)).float().to(device)
#t = t[batch.batch]

#microbatch = 14

#for j, batch in enumerate(train_loader):
    #num_graphs = len(batch.ptr) - 1
    #for i in range(0, num_graphs // microbatch * microbatch, microbatch): 
        #micro = batch[i : i + microbatch]
        #micro_batch = Batch.from_data_list(micro)
     

#m_index = torch.arange(micro_batch.x.size(0))
#print(m_index)   


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss

    PH = enum.auto() # Pseudo-Huber
    PL = enum.auto() # Pseudo-LPIPS
