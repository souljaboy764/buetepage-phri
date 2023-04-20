import torch
import torch.nn.functional as F

import numpy as np
import argparse, os

import networks

parser = argparse.ArgumentParser(description='Buetepage HRI Dynamics Training')
# parser.add_argument('--human-ckpt', type=str, metavar='HUMAN-CKPT', default='logs/vae_hh_orig_oldcommit_AdamW_07011535_tdmfixed/models/final.pth',
parser.add_argument('--human-ckpt', type=str, metavar='HUMAN-CKPT', default='logs/vae_hh_orig_oldcommit_AdamW_07011535/models/final.pth',
					help='Path to the Human dynamics checkpoint.')
parser.add_argument('--robot-ckpt', type=str, metavar='ROBOT-CKPT', default='logs/vae_hr_AdamW_07031331/models/final.pth',
					help='Path to the VAE checkpoint, where the TDM models will also be saved.')
parser.add_argument('--src', type=str, default='./data/orig_hr/tdm_data.npz', metavar='DATA',
					help='Path to read training and testin data (default: ./data/orig_hr/tdm_data.npz).')
args = parser.parse_args()
torch.manual_seed(128542)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = []

for ckpt in [args.robot_ckpt, args.human_ckpt]:
	if not os.path.exists(ckpt):
		print('VAE MODEL NOT FOUND')
		exit(-1)
	DEFAULT_RESULTS_FOLDER = os.path.dirname(os.path.dirname(ckpt))
	VAE_MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	vae_args = vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	vae_config = vae_hyperparams['vae_config'].item()
	models.append(getattr(networks, vae_args.model)(**(vae_config.__dict__)).to(device))
	ckpt = torch.load(ckpt)
	models[-1].load_state_dict(ckpt['model'])
	models[-1].eval()

robot_vae = models[0]
human_vae = models[1]

print("Reading Data")
h_vae_idx = np.arange(480)
r_vae_idx = np.arange(280) + 480
with np.load(args.src, allow_pickle=True) as data:
	test_data = data['test_data']

print("Starting Evaluation")
total_loss = []
xr_gen = []
xh_gen = []
for i, x in enumerate(test_data):
	seq_len, dims = x.shape
	x = torch.Tensor(x).to(device)
	xh_gt = x[:,h_vae_idx].to(device)
	xr_gt = x[:,r_vae_idx].to(device)
	with torch.no_grad():
		xh_gen_i, _, _ = human_vae(xh_gt)
		xr_gen_i, _, _ = robot_vae(xr_gt)
				
	loss = F.mse_loss(xr_gt, xr_gen_i, reduction='none')
	total_loss.append(loss.detach().cpu().numpy())
	xr_gen.append(xr_gen_i.detach().cpu().numpy())
	xh_gen.append(xh_gen_i.detach().cpu().numpy())

xr_gen = np.array(xr_gen)
xh_gen = np.array(xh_gen)
np.savez_compressed('hri_vae_test.npz', x_gen=xr_gen, test_data=test_data)