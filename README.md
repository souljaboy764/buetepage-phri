# Bütepage PHRI

Pytorch Implementation of Bütepage et al. (2020) "Imitating by generating: Deep generative models for imitation of interactive tasks." Frontiers in Robotics and AI.

## Requirements

Install the requirements in [`requirements.txt`](requirements.txt) by running

```bash
pip install -r requirements.txt
```

Clone the repo `https://github.com/souljaboy764/phd_utils` and follow the installation instructions in its README.

## Training

### Human-Human Interactions

1. Train the Human VAE:

    ```bash
    python train_vae.py --model BP_HH --resuts /path/to/human_vae_ckpt
    ```

    (replace `BP_HH` with `NUISI_HH` or `ALAP` for the respective datasets)

2. Train the Human TDM:

    ```bash
    python train_vae.py --vae-ckpt /path/to/human_vae_ckpt/models/XXXX.pth
    ```

    Here, `/path/to/human_vae_ckpt/models/XXXX.pth` is the path to the human VAE checkpoint to use. The TDM checkpoints will be saved in `/path/to/human_vae_ckpt/tdm/models/` as `tdm_XXXX.pth` where `XXXX` is the epoch number.

### Human-Robot Interactions

1. Train the Robot VAE:

    ```bash
    python train_vae.py --model BP_PEPPER --resuts /path/to/robot_vae_ckpt
    ```

    (replace `BP_PEPPER` with `NUISI_PEPPER` or `BP_YUMI` for the respective datasets)

2. Train the HRI Dynamics model:

    ```bash
    python train_hri.py --human-ckpt /path/to/human_vae_ckpt/tdm/models/tdm_XXXX.pth --robot-ckpt /path/to/robot_vae_ckpt/models/XXXX.pth
    ```

    where `/path/to/human_vae_ckpt/tdm/models/tdm_XXXX.pth` is the path to the Human TDM checkpoint and `/path/to/robot_vae_ckpt/models/XXXX.pth` is the path to the Robot VAE checkpoint. The HRI dynamics checkpoints will be saved in `/path/to/robot_vae_ckpt/dynamics/models/` as `hri_XXXX.pth` where `XXXX` is the epoch number.

## Testing

The output of the below testing codes is the Mean squared prediction error and standard deviation for each interaction in the dataset of the model that is being evaluated.

### Human-Human Interactions

```bash
python test_tdm.py --tdm-ckpt /path/to/human_vae_ckpt/tdm/models/tdm_XXXX.pth
```
