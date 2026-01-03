import argparse

from model.Graph_Model_ import GraphModel
from utils.gaussian_diffusion_ import (
    VP_Diffusion,
    get_named_beta_schedule,
    LossType,
)

def model_and_diffusion_defaults():
    """
    Defaults for graph-based SSM diffusion training (VP-DDDM).
    """
    return dict(
        model_type="jamba",  # {"jamba", "mamba", "mamba_2", "hydra", "transformer"}
        channels=512,
        depth=6,
        heads=4,
        headdim=16,
        num_layers=8,
        attn_dropout=0.25,
        in_node_nf=512,
        out_node_nf=512,
        in_edge_nf=512,
        hidden_nf=512,
        pe_dim=20,
        context_dim=16,
        time_embed_dim=16,
        d_state=512,
        d_conv=256,
        order_by_degree=True,
        shuffle_ind=0,
        num_tokens=2000,
        num_graphs=55000,
        diffusion_steps=1000,
        noise_schedule="cosine",     # {"linear", "cosine"}
        beta_start=1e-5,
        beta_end=2e-2,
        loss_type="ph",              # {"mse", "ph"}
        c=0.000069,
        use_checkpoint=False,
    )


def data_defaults():
    """
    Defaults for graph dataset & dataloader.
    """
    return dict(
        data_dir="",
        split_path="",
        cache="",
        dataset="qm9",          # {"qm9", "drugs"}
        num_workers=4,
        limit_train_mols=None,
    )


def create_model_and_diffusion(**kwargs):
    """
    Create graph model and VP diffusion from keyword args.
    """
    hparams = argparse.Namespace(**kwargs)

    model = GraphModel(hparams)
    diffusion = create_vp_diffusion(hparams)

    return model, diffusion


def create_vp_diffusion(hparams):
    loss_type_str = hparams.loss_type.lower()

    if loss_type_str == "mse":
        loss_type = LossType.MSE
    elif loss_type_str == "ph":
        loss_type = LossType.PH
    else:
        raise ValueError(f"Unknown loss_type: {hparams.loss_type}")

    betas = get_named_beta_schedule(
        schedule_name=hparams.noise_schedule,
        num_diffusion_timesteps=hparams.diffusion_steps,
        beta_start=hparams.beta_start,
        beta_end=hparams.beta_end,
    )

    return VP_Diffusion(
        betas=betas,
        loss_type=loss_type,
        c=hparams.c,
    )
