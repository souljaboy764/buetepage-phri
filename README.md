# Bütepage PHRI

Pytorch Implementation of Bütepage et al. (2020) "Imitating by generating: Deep generative models for imitation of interactive tasks." Frontiers in Robotics and AI.

## Requirements

Install the requirements in [`requirements.txt`](requirements.txt) by running

```bash
pip install -r requirements.txt
```

Clone the repository [`https://github.com/jbutepage/human_robot_interaction_data`](https://github.com/jbutepage/human_robot_interaction_data) for the data from the paper

## Data Preprocessing

To preprocess the data according to the paper, run:

```bash
python data_preproc.py --dst-dir /path/to/preproc_data
```

This creates three files, `labelled_sequences.npz`, `vae_data.npz` and `tdm_data.npz`.

## Running Training

Once the data preprocessing is done, you can train the VAE by running:

```bash
python train_vae.py --src /path/to/preproc_data/vae_data.npz (--results /path/to/results_dir)
```

To visualise the resutls of the training, run:

```bash
python test_hh_vae.py --src /path/to/preproc_data/vae_data.npz --ckpt /path/to/model.pth
```

## TODO

- Update Code and README for running TDM
- Update Code and README for running HRI
