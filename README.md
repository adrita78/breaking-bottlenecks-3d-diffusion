# breaking-bottlenecks-3d-diffusion

<p align="center">
  <img src="assets/Fiigures.png" width="700">
</p>

<p>
<strong>Overview.</strong>
This project studies diffusion-based generative modeling through the lens of
forward noising and reverse-time denoising dynamics.
The visualization above summarizes the full generative pipeline.

In the forward process (top), data samples drawn from the empirical distribution
\(p_{\text{data}}\) are progressively corrupted with noise until they converge
to a simple prior distribution \(p_{\text{prior}}\).
During training, a neural denoising model learns to approximate the corresponding
reverse-time transition kernels that undo this corruption.

In the generative phase (bottom), new samples are obtained by initializing from
pure noise \(x_T \sim p_{\text{prior}}\) and iteratively denoising back to a clean
sample \(x_0\).
This framework enables efficient learning of complex data distributions while
maintaining a tractable and stable sampling procedure.
</p>

