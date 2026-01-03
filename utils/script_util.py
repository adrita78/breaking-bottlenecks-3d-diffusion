import argparse
import inspect

from model.Graph_Model_ import GraphModel
from utils.gaussian_diffusion_ import VP_Diffusion, get_named_beta_schedule, LossType


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
        loss_type="ph",             # {"mse", "ph"}
        c=c= 0.000069,
        schedule_sampler="uniform",
        use_checkpoint=False,
    )

def create_model_and_diffusion(hparams):

    if isinstance(hparams, dict):
        hparams = argparse.Namespace(**hparams)

    model = create_graph_model(hparams)
    diffusion = create_vp_diffusion(hparams)

    return model, diffusion

def create_graph_model(hparams):
    
    return GraphModel(hparams)

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

    diffusion = VP_Diffusion(
        betas=betas,
        loss_type=loss_type,
        c=hparams.c,
    )

    return diffusion
    
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

