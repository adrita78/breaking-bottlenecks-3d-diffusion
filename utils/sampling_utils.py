import random
from utils.featurization import featurize_mol, featurize_mol_from_smiles
import torch, copy
from copy import deepcopy
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

from utils.utils import time_limit, TimeoutException
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

def embed_seeds(
    mol,
    data,
    n_confs,
    single_conf=False,
    smi=None,
    embed_func=None,
    seed_confs=None,
    pdb=None,
    mmff=False,
):
    
    # --------------------------------------------------
    # 1. Embed conformers (RDKit only)
    # --------------------------------------------------
    embed_num_confs = n_confs if not single_conf else 1

    try:
        mol = embed_func(mol, embed_num_confs)
    except Exception as e:
        print(e)
        return [], None

    if mol.GetNumConformers() != embed_num_confs:
        print(mol.GetNumConformers(), '!=', embed_num_confs)
        return [], None

    if mmff:
        try_mmff(mol)

    # --------------------------------------------------
    # 2. Optional PDB logging (unchanged)
    # --------------------------------------------------
    if pdb:
        pdb = PDBFile(mol)

    conformers = []

    # --------------------------------------------------
    # 3. Generate conformer Data objects
    # --------------------------------------------------
    for i in range(n_confs):
        data_conf = copy.deepcopy(data)

        # Always use the i-th conformer directly
        seed_mol = copy.deepcopy(mol)
        if seed_mol.GetNumConformers() > 1:
            for j in range(seed_mol.GetNumConformers()):
                if j != i:
                    seed_mol.RemoveConformer(j)

        data_conf.pos = torch.from_numpy(
            seed_mol.GetConformers()[0].GetPositions()
        ).float()

        data_conf.seed_mol = copy.deepcopy(seed_mol)

        if pdb:
            pdb.add(data_conf.pos, part=i, order=0)
            pdb.add(torch.zeros_like(data_conf.pos), part=i, order=-1)

        conformers.append(data_conf)

    # --------------------------------------------------
    # 4. Cleanup RDKit molecule (keep first conformer)
    # --------------------------------------------------
    if mol.GetNumConformers() > 1:
        for j in range(1, mol.GetNumConformers()):
            mol.RemoveConformer(j)

    return conformers, pdb


def pyg_to_mol(
    mol,
    data,
    mmff=False,
    rmsd=False,
    copy=True
):

    # --------------------------------------------------
    # 1. Ensure a conformer exists
    # --------------------------------------------------
    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf)

    # --------------------------------------------------
    # 2. Write coordinates from data.pos
    # --------------------------------------------------
    coords = data.pos
    if not isinstance(coords, np.ndarray):
        coords = coords.detach().cpu().double().numpy()

    conf = mol.GetConformer(0)
    for i in range(coords.shape[0]):
        conf.SetAtomPosition(
            i,
            Point3D(
                float(coords[i, 0]),
                float(coords[i, 1]),
                float(coords[i, 2]),
            )
        )

    # --------------------------------------------------
    # 3. Optional MMFF relaxation
    # --------------------------------------------------
    if mmff:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass

    # --------------------------------------------------
    # 4. Optional RMSD (only if seed exists)
    # --------------------------------------------------
    if rmsd and hasattr(data, "seed_mol"):
        try:
            mol.rmsd = AllChem.GetBestRMS(
                Chem.RemoveHs(data.seed_mol),
                Chem.RemoveHs(mol),
            )
        except Exception:
            pass

    if not copy:
        return mol

    return deepcopy(mol)

class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
