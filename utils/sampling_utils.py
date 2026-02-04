import random
from utils.featurization import featurize_mol, featurize_mol_from_smiles
import torch, copy
from copy import deepcopy
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

#from utils.utils import time_limit, TimeoutException
from utils.visualise import PDBFile
from spyrmsd import molecule, graph
from rdkit.Geometry import Point3D
from copy import deepcopy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def try_mmff(mol):
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        return True
    except Exception as e:
        return False
    
def get_seed(smi, seed_confs=None, dataset='drugs'):
    if seed_confs:
        if smi not in seed_confs:
            print("smile not in seeds", smi)
            return None, None
        mol = seed_confs[smi][0]
        data = featurize_mol(mol, dataset)
    else:
        mol, data = featurize_mol_from_smiles(smi, dataset=dataset)
        if not mol:
            return None, None

    return mol, data



def embed_seeds(mol, data, n_confs, single_conf=False, smi=None, embed_func=None, seed_confs=None, pdb=None, mmff=False):
    if not seed_confs:
        embed_num_confs = n_confs if not single_conf else 1
        try:
            mol = embed_func(mol, embed_num_confs)
        except Exception as e:
            print(e.output)
            pass
        if len(mol.GetConformers()) != embed_num_confs:
            print(len(mol.GetConformers()), '!=', embed_num_confs)
            return [], None
        if mmff: try_mmff(mol)

    if pdb: pdb = PDBFile(mol)
    conformers = []
    for i in range(n_confs):
        data_conf = copy.deepcopy(data)
        if single_conf:
            seed_mol = copy.deepcopy(mol)
        elif seed_confs:
            seed_mol = random.choice(seed_confs[smi])
        else:
            seed_mol = copy.deepcopy(mol)
            [seed_mol.RemoveConformer(j) for j in range(n_confs) if j != i]

        data_conf.pos = torch.from_numpy(seed_mol.GetConformers()[0].GetPositions()).float()
        data_conf.seed_mol = copy.deepcopy(seed_mol)
        if pdb:
            pdb.add(data_conf.pos, part=i, order=0, repeat=still_frames)
            if seed_confs:
                pdb.add(data_conf.pos, part=i, order=-2, repeat=still_frames)
            pdb.add(torch.zeros_like(data_conf.pos), part=i, order=-1)

        conformers.append(data_conf)
    if mol.GetNumConformers() > 1:
        [mol.RemoveConformer(j) for j in range(n_confs) if j != 0]
    return conformers, pdb
       
    
def pyg_to_mol(mol, data, mmff=False, rmsd=True, copy=True):
    
    if not mol.GetNumConformers():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conformer)

    # Get coordinates
    coords = data.pos
    if not isinstance(coords, np.ndarray):
        coords = coords.double().cpu().numpy()

    # Set atomic positions
    conf = mol.GetConformer(0)
    for i in range(coords.shape[0]):
        conf.SetAtomPosition(
            i, Point3D(coords[i, 0], coords[i, 1], coords[i, 2])
        )

    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(
                mol, mmffVariant="MMFF94s"
            )
        except Exception:
            pass


    if rmsd and hasattr(data, "seed_mol"):
        try:
            mol.rmsd = AllChem.GetBestRMS(
                Chem.RemoveHs(data.seed_mol),
                Chem.RemoveHs(mol)
            )
        except Exception:
            pass

    return mol if not copy else deepcopy(mol)


