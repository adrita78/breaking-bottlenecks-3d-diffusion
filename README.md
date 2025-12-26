# breaking-bottlenecks-3d-diffusion

<p align="center">
  <img src="assets/Fiigures.png" width="700">
</p>

<p>
<strong>Overview.</strong>
This repository introduces a fast and scalable diffusion framework for 3D molecular generation based on deterministic denoising. By reinterpreting Directly Denoising Diffusion Models (DDDM) through the Reverse Transition Kernel (RTK) framework, the approach unifies deterministic and stochastic diffusion under a single probabilistic view, enabling variance-free sampling, improved numerical stability, and significantly faster inference. An SE(3)-equivariant, state-space–model (SSM)–based architecture is used to efficiently capture long-range dependencies in large molecular graphs.

**Key highlights**

- Deterministic, RTK-guided denoising for fast and stable molecular diffusion
- SE(3)-equivariant SSM architecture enabling scalable 3D generation on large molecules
</p>

