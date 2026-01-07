import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import torch.nn.functional as F
import numpy as np
import torch as th
import enum
from abc import ABC, abstractmethod
import torch.distributed as dist
from loss_functions import ph_loss, mean_flat

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps,beta_start,beta_end):
    """
    scheduler for VP model
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.

        beta_start = beta_start / num_diffusion_timesteps
        beta_end = beta_end / num_diffusion_timesteps
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss

    PH = enum.auto() # Pseudo-Huber


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from `arr` based on `timesteps`, expanding to `broadcast_shape`.
    """
    if timesteps.ndimension() == 0:
        timesteps = timesteps.unsqueeze(0)

    if len(broadcast_shape) > 1:
        broadcast_shape = (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1)
    arr = torch.tensor(arr, dtype=torch.float32)
    try:
        extracted = arr[timesteps.long()]
        if extracted.numel() == 1:
            extracted = extracted.view(1)
        else:
            extracted = extracted.view(*broadcast_shape)
    except Exception as e:
        raise e

    return extracted    

def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class VP_Diffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    
    """

    def __init__(
        self,
        betas,
        loss_type,
        c = 0.0,
             
    ):   
        self.loss_type = loss_type
        self.c = c
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas 
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        

    def q_sample(self, x, t, noise=None):
        x = x.float()

        if noise is None:
            noise = torch.randn_like(x)
        assert noise.shape == x.shape, f"Noise shape {noise.shape} does not match x_start shape {x.shape}."
        broadcast_shape = x.shape

        sqrt_alphas_tensor = _extract_into_tensor(self.sqrt_alphas_cumprod, t, broadcast_shape)
        sqrt_one_minus_alphas_tensor = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, broadcast_shape)

        noisy_x = (
        sqrt_alphas_tensor * x +
        sqrt_one_minus_alphas_tensor * noise
    )
        return noisy_x    

    def marginal_prob(self, x, t):
        """
        Compute the marginal mean and std.
        """
        log_mean_coeff = -0.5 * t ** 2 * (self.beta_max - self.beta_start) - 0.5 * t * self.beta_start
        mean = th.exp(log_mean_coeff[:, None, None]) * x
        std = th.sqrt(1.0 - th.exp(2.0 * log_mean_coeff))
        return mean, std    
    
    def add_noise(self, batch, timesteps, device):

        num_graphs = batch.ptr.size(0) - 1
        
        noisy_x = []
        for i in range(num_graphs):
            start, end = batch.ptr[i], batch.ptr[i + 1]
            x = batch.x[start:end]
            t = timesteps[i].clone().detach()
            t = torch.tensor(t, dtype=torch.float32).to(device)
            x_noisy = self.q_sample(x, t)
            noisy_x.append(x_noisy)

        batch.x = torch.cat(noisy_x, dim=0)
        return batch

    def training_losses(self,
        model,
        batch,
        timesteps,
        index,
        condition,
        c: float = 0.0,
        device=None,
        model_kwargs=None
    ):
        """
        Compute the diffusion training losses.

        :param model: the model to evaluate.
        :param batch: a batch of data.
        :param timesteps: a 1-D tensor of timesteps.
        :param index: a 1-D tensor of indices for the timesteps.
        :param condition: conditioning information for the model.
        :param c: Pseudo-Huber loss parameter.
        :param device: the torch device.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
                             pass to the model.
        :return: a dict with keys "guide" and "iter", containing the
                 corresponding losses.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if device is None:
            device = next(model.parameters()).device
        batch = batch.to(device)
        timesteps = timesteps.to(device)
        condition = condition.to(device) if condition is not None else None

        # Apply forward diffusion noise
        batch = self.add_noise(batch, timesteps, device)

        terms = {}

        # Graph-level representation
        x_graph = global_mean_pool(batch.x, batch.batch)

        # Model forward pass
        model_output = model(
           t=timesteps,
           context=condition,
           batch=batch,
           device=device,
           **model_kwargs,
        )

        model_output = x_graph - model_output

        if self.loss_type == LossType.MSE:
            terms["guide"] = mean_flat((model_output - x_graph) ** 2)
            terms["iter"] = mean_flat((model_output - condition) ** 2)

        # Optional buffer update
        # model.module.update_xbar(model_output, index)

        elif self.loss_type == LossType.PH:
            terms["guide"] = ph_loss(model_output, x_graph, c)
            terms["iter"] = ph_loss(model_output, condition, c)

        else:
            raise NotImplementedError(self.loss_type)

        return terms, model_output


    def p_sample(self, model, batch, t, x_bar,model_kwargs=None):
        x_graph = global_mean_pool(batch.x, batch.batch)
        sample = x_graph - model(batch, t, context=x_bar,**model_kwargs)
        return sample

    def p_sample_loop(
        self,
        model,
        batch,
        noise=None,
        condition=None,
        model_kwargs=None,
        device=None,
        sample_steps=100,
        
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        batch = batch.to(device)
        num_graphs= (batch.ptr.shape[0]-1)
        
        if noise is None:
            x_T = torch.randn_like(batch.x, device=device)
        else:
            x_T = noise.to(device)
        batch.x = x_T    
        if condition is None:
            x_bar = torch.zeros(num_graphs, 74, device=device)
        else:
            x_bar = condition.to(device)

        T = torch.full((num_graphs,), self.num_timesteps, device=device, dtype=torch.long)

        for _ in range(sample_steps):
            with th.no_grad():
                out = self.p_sample(
                    model,
                    batch,
                    T,
                    x_bar,
                    model_kwargs=model_kwargs,
                )
                x_bar = out
                batch.x = x_bar[batch.batch]
               
        return batch
        


