import argparse
import os
import torch as th
import torch.distributed as dist

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from utils import dist_util, logger
from utils.featurization import construct_loader
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import argparse
import os
import copy
import pickle

import torch as th
import torch.distributed as dist

from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem

from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem

from utils import dist_util, logger
from utils.featurization import construct_loader
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# -------------------------------------------------------
# RDKit conversion utilities
# -------------------------------------------------------

def pyg_graph_to_conformer(data, do_mmff=True):
    """
    Convert a PyG Data object to an RDKit molecule with a single conformer.

    Required attributes in `data`:
      - data.mol : RDKit Mol (seed molecule, no or dummy conformer)
      - data.pos : (N, 3) tensor with generated coordinates
    """
    mol = copy.deepcopy(data.mol)

    conf = Chem.Conformer(mol.GetNumAtoms())
    pos = data.pos.detach().cpu().numpy()

    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*pos[i]))

    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    if do_mmff:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass

    return mol


def batch_to_conformers(batch, do_mmff=True):
    """
    Convert a PyG Batch into a list of RDKit molecules.
    """
    mols = []
    for data in batch.to_data_list():
        mol = pyg_graph_to_conformer(data, do_mmff=do_mmff)
        mols.append(mol)
    return mols


# -------------------------------------------------------
# Main sampling logic
# -------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    # Distributed setup
    dist_util.setup_dist()
    logger.configure()

    device = dist_util.dev()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()

    # Validation loader (used as seed molecules)
    _, val_loader = construct_loader(
        data_dir=args.data_dir,
        split_path=args.split_path,
        cache=args.cache,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_train_mols=args.limit_train_mols,
        modes=("train", "val"),
    )

    logger.log("Sampling conformers...")

    all_mols = []

    while len(all_mols) < args.num_samples:
        for batch in val_loader:
            batch = batch.to(device)

            with th.no_grad():
                sampled_batch = diffusion.p_sample_loop(
                    model=model,
                    batch=batch,
                    sample_steps=args.sample_steps,
                )

            # Convert batch → RDKit conformers
            sampled_mols = batch_to_conformers(
                sampled_batch,
                do_mmff=not args.no_mmff,
            )

            # Gather across GPUs
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, sampled_mols)

            for rank_mols in gathered:
                all_mols.extend(rank_mols)

            logger.log(f"Generated {len(all_mols)} conformers")

            if len(all_mols) >= args.num_samples:
                break

    # -------------------------------------------------------
    # Save results (rank 0 only)
    # -------------------------------------------------------
    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), "sampled_conformers.pkl")
        logger.log(f"Saving conformers to {out_path}")

        with open(out_path, "wb") as f:
            pickle.dump(all_mols[: args.num_samples], f)

    dist.barrier()
    logger.log("Sampling complete")


# -------------------------------------------------------
# Argument parser
# -------------------------------------------------------

def create_argparser():
    defaults = dict(
        num_samples=1000,
        batch_size=16,
        model_path="",
        sample_steps=100,
        no_mmff=False,
    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
