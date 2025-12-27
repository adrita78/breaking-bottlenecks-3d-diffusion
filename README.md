# breaking-bottlenecks-3d-diffusion

<p align="center">
  <img src="assets/Fiigures.png" width="700">
</p>

<p>
<strong>Overview.</strong>
This project introduces a fast and scalable diffusion framework for 3D molecular generation based on deterministic denoising. By reinterpreting Directly Denoising Diffusion Models (DDDM) through the Reverse Transition Kernel (RTK) framework, the approach unifies deterministic and stochastic diffusion under a single probabilistic view, enabling variance-free sampling, improved numerical stability, and significantly faster inference. An SE(3)-equivariant, state-space–model (SSM)–based architecture is used to efficiently capture long-range dependencies in large molecular graphs.

**Key highlights**

- Deterministic, RTK-guided denoising for fast and stable molecular diffusion
- SE(3)-equivariant SSM architecture enabling scalable 3D generation on large molecules
</p>

## Installation

### Step 1: Create a Conda Environment

We recommend using **Conda** to manage dependencies. If Conda is not installed, refer to the  
[Anaconda installation guide](https://docs.anaconda.com/anaconda/install/).

Create a Python 3.10 environment:

```bash
conda create --name my_env python=3.10
```
Activate the environment:

conda activate my_env


