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

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

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

    logger.log("sampling molecules...")

    all_batches = []

    while len(all_batches) * args.batch_size < args.num_samples:
        for batch in val_loader:
            batch = batch.to(dist_util.dev())

            with th.no_grad():
                sampled_batch = diffusion.p_sample_loop(
                    model=model,
                    batch=batch,
                    sample_steps=args.sample_steps,
                )

            # Gather across GPUs
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, sampled_batch)

            all_batches.extend(gathered)

            logger.log(
                f"generated {len(all_batches) * args.batch_size} molecules"
            )

            if len(all_batches) * args.batch_size >= args.num_samples:
                break

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), "sampled_molecules.pt")
        logger.log(f"saving samples to {out_path}")

        th.save(all_batches[: args.num_samples], out_path)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        num_samples=10000,
        batch_size=16,
        model_path="",
        sample_steps=100,
    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
