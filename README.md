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
```bash
conda activate my_env
```
### Step 2: Install Dependencies
Install PyTorch (with CUDA support)

CUDA acceleration is required to run this project. You must have an NVIDIA GPU and install PyTorch with CUDA 12.1 support:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```
### Step 3: Install State Space Model Dependencies

This project relies on state space model (SSM) libraries such as **Mamba** and **Jamba**.

Install the core packages using `pip`:
```bash
pip install mamba-ssm
pip install jamba
```
Install Remaining Dependencies
Install the remaining required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
## Downloading the Dataset
You can download the dataset directly from Harvard Dataverse and extract it using the following commands.
### Step 1: Download the data
```bash
wget https://dataverse.harvard.edu/api/access/datafile/4327252 -O dataset.tar
```
This downloads the dataset as a tar archive.
### Step 2: Extract the archive
```bash
tar -xvf dataset.tar
```
This will extract the dataset into the current directory.

Additional Notes:
- Ensure `wget` is installed in your system.
- The extracted files may take several GB of disk space.
- If you are on a cluster, run this on a login node or a data-transfer node.
## Dataset Setup
```bash
# Create dataset directories
mkdir -p data/drugs/raw
mkdir -p data/drugs/processed
mkdir -p splits
```
Place your dataset files and split file as follows:
```bash
data/drugs/raw/           
data/drugs/processed/     
splits/drugs_split.npy   # train/val split file
```
## 🚀 Training Command
```bash
torchrun \
--standalone \
--nproc_per_node=8 \
./scripts/train.py \
  --data_dir data/drugs \
  --split_path splits/drugs_split.npy \
  --dataset drugs \
  --batch_size 32 \
  --num_workers 4 \
  --noise_schedule cosine \
  --diffusion_steps 1000 \
  --lr 1e-4
```
## 🔁 Resume Training from Checkpoint

To resume training from a previously saved checkpoint, run:
```bash
torchrun \
--standalone \
--nproc_per_node=8 \
./scripts/train.py \
  --data_dir data/drugs \
  --split_path splits/drugs_split.npy \
  --resume_checkpoint checkpoints/model.pt
```
