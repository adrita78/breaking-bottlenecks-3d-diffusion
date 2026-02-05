import random
import os
import pandas as pd
from utils.featurize_mol import featurize_mol, featurize_mol_from_smiles
import torch, copy
from copy import deepcopy
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Geometry import Point3D

#from utils.utils import time_limit, TimeoutException
from visualise import PDBFile
from spyrmsd import molecule, graph
from rdkit.Geometry import Point3D
from copy import deepcopy
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
import random
import torch
import yaml
from utils.LapPE import AddCustomLaplacianEigenPE
from model.Graph_Model_ import GraphModel
from utils.sampling_utils import get_seed, embed_seeds, pyg_to_mol

dataset = "qm9"
parser = ArgumentParser()
                  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--ckpt', type=str, default='model0132.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--out', type=str, help='Path to the output pickle file')
parser.add_argument('--test_csv', type=str, help='Path to csv file with list of smiles and number conformers')
parser.add_argument('--pre_mmff', action='store_true', default=False, help='Whether to run MMFF on the local structure conformer')
parser.add_argument('--post_mmff', action='store_true', default=False, help='Whether to run MMFF on the final generated structures')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--seed_confs', help='Path to directly specify the seed conformers')
parser.add_argument('--seed_mols', help='Path to directly specify the seed molecules (instead of from SMILE)')
parser.add_argument('--single_conf', action='store_true', default=False, help='Whether to start from a single local structure')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--limit_mols', type=int, default=None, help='Limit to the number of molecules')
parser.add_argument('--confs_per_mol', type=int, default=None, help='If set for every molecule this number of conformers is generated, '
                                                                    'otherwise 2x the number in the csv file')
                                                                    
                                                                  
parser.add_argument('--dataset', type=str, default='qm9', help='Dataset name (e.g., qm9, drugs)')                    
parser.add_argument("--config", type=str, default="model_config.yaml")  
parser.add_argument('--batch_size', type=int, default=32, help='Number of conformers generated in parallel')
parser.add_argument('--dump_pymol', type=str, default=None, help='Whether to save .pdb file with denoising dynamics')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
                                                                                                                                    
                                                                    
args = parser.parse_args()

if args.seed_confs:
    print("Using local structures from", args.seed_confs)
    with open(args.seed_confs, 'rb') as f:
        seed_confs = pickle.load(f)
elif args.seed_mols:
    print("Using molecules from", args.seed_mols)
    with open(args.seed_mols, 'rb') as f:
        seed_confs = pickle.load(f)

def try_mmff(mol):
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        return True
    except Exception as e:
        return False
    
    
def embed_func(mol, numConfs):
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=5)
    return mol    
    
    
test_data = pd.read_csv(args.test_csv).values
if args.limit_mols:
    test_data = test_data[:args.limit_mols]

conformer_dict = {}
if args.tqdm:
    test_data = tqdm(enumerate(test_data), total=len(test_data))
else:
    test_data = enumerate(test_data)


def sample_confs(raw_smi, n_confs, smi):
    print(raw_smi)
    if args.seed_confs:
        mol, data = get_seed(raw_smi, seed_confs=seed_confs, dataset=args.dataset)
    elif args.seed_mols:
        mol, data = get_seed(smi, seed_confs=seed_confs, dataset=args.dataset)
        mol.RemoveAllConformers()
    else:
        mol, data = get_seed(smi, dataset=args.dataset)
    if not mol:
        print('Failed to get seed', smi)
        return None

    if args.seed_confs:
        conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=args.single_conf, smi=raw_smi,
                                      pdb=args.dump_pymol, seed_confs=seed_confs)
    else:
        conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=args.single_conf,
                                      pdb=args.dump_pymol, embed_func=embed_func, mmff=args.pre_mmff)
    if not conformers:
        print("Failed to embed", smi)
        return None

    sampled_batch = diffusion.p_sample_loop(
                    model=model,
                    conformers= conformers,
                    sample_steps=args.sample_steps,)


