from argparse import ArgumentParser
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import random
import torch
import yaml
from model.Graph_Model_ import GraphModel
from utils.featurization import featurize_mol_from_smiles
from torch_geometric.data import Batch
#from model.inference import construct_conformers

parser = ArgumentParser()
parser.add_argument('--trained_model_dir', type=str, default="trained_model/qm9_model")
parser.add_argument('--out', type=str, default="generated_confs.pkl")
parser.add_argument('--test_csv', type=str, default="data/qm9/test.csv")
parser.add_argument('--dataset', type=str, default='qm9')
parser.add_argument('--mmff', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

trained_model_dir = args.trained_model_dir
test_csv = args.test_csv
dataset = args.dataset
mmff = args.mmff

with open(f'{trained_model_dir}/model_parameters.yml') as f:
    model_parameters = yaml.full_load(f)
model = GraphModel(**model_parameters)

state_dict = torch.load(f'{trained_model_dir}/best_model.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model.eval()

test_data = pd.read_csv(test_csv)

conformer_dict = {}
for smi, n_confs in tqdm(test_data.values):
    
    # create data object (skip smiles rdkit can't handle)
    tg_data = featurize_mol_from_smiles(smi, dataset=dataset)
    if not tg_data:
        print(f'failed to featurize SMILES: {smi}')
        continue
    data = Batch.from_data_list([tg_data])
