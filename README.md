# Bütepage PHRI

Pytorch Implementation of Bütepage et al. (2020) "Imitating by generating: Deep generative models for imitation of interactive tasks." Frontiers in Robotics and AI.

## Requirements

Install the requirements in [`requirements.txt`](requirements.txt) by running

```bash
pip install -r requirements.txt
```

Clone the repo `https://github.com/souljaboy764/phd_utils` and follow the installation instructions in its README.

## Running Training

The steps for training on the Human-Human Interactions is:

1. Train the Human VAE `python train_vae.py --model HH` (replace `HH` with `NUISI_HH` or `ALAP` for the respective datasets)
2. Train the Human TDM model `python train_vae.py --vae-ckpt /path/to/vae_ckpt` where `/path/to/vae_ckpt` is the path to the VAE checkpoint.
