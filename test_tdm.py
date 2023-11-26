import torch

import numpy as np
import os, argparse

import networks
from config import *

from phd_utils.dataloaders import *

parser = argparse.ArgumentParser(description='Buetepage et al. (2020) Human-Human Interaction Testing')
parser.add_argument('--tdm-ckpt', type=str, metavar='CKPT', required=True,
					help='Path to the TDM checkpoint to test.')
args = parser.parse_args()
torch.manual_seed(128542)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tdm_models_folder = os.path.dirname(args.tdm_ckpt)

if os.path.exists(os.path.join(tdm_models_folder,'tdm_hyperparams.npz')):
	# print(os.path.join(tdm_models_folder,'tdm_hyperparams.npz'))
	hyperparams = np.load(os.path.join(tdm_models_folder,'tdm_hyperparams.npz'), allow_pickle=True)
	tdm_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	tdm_config = hyperparams['tdm_config'].item()
else:
	print(os.path.join(tdm_models_folder,'tdm_hyperparams.npz'))
	print('No TDM configs found')
	exit(-1)

vae_hyperparams = np.load(os.path.join(os.path.dirname(os.path.dirname(tdm_models_folder)),'models', 'hyperparams.npz'), allow_pickle=True)
vae_args = vae_hyperparams['args'].item()
vae_config = vae_hyperparams['config'].item()
config = vae_hyperparams['global_config'].item()

tdm_1 = networks.TDM(**(tdm_config.__dict__)).to(device)
tdm_2 = networks.TDM(**(tdm_config.__dict__)).to(device)

ckpt = torch.load(args.tdm_ckpt)
tdm_1.load_state_dict(ckpt['model_1'])
tdm_2.load_state_dict(ckpt['model_2'])
global_step = ckpt['epoch']

vae = networks.VAE(**(vae_config.__dict__)).to(device)
ckpt = torch.load(tdm_args.vae_ckpt)
vae.load_state_dict(ckpt['model'])
vae.eval()

if vae_args.model =='BP_HH':
	dataset = buetepage.HHWindowDataset
elif vae_args.model =='NUISI_HH':
	dataset = nuisi.HHWindowDataset
elif vae_args.model =='ALAP':
	dataset = alap.HHWindowDataset
test_dataset = dataset(train=False, window_length=config.WINDOW_LEN, downsample=0.2)

test_dataset.labels = []
for idx in range(len(test_dataset.actidx)):
	for i in range(test_dataset.actidx[idx][0], test_dataset.actidx[idx][1]):
		label = np.zeros((test_dataset.traj_data[i].shape[0], len(test_dataset.actidx)))
		label[:, idx] = 1
		test_dataset.labels.append(label)


if vae_args.model == 'BP_HH' or vae_args.model == 'NUISI_HH':
	tdm_config = human_tdm_config()
	p1_tdm_idx = np.concatenate([np.arange(18),np.arange(-4,0)])
	p2_tdm_idx = np.concatenate([90+np.arange(18),np.arange(-4,0)])
	p1_vae_idx = np.arange(90)
	p2_vae_idx = np.arange(90) + 90
elif vae_args.model == 'ALAP':
	tdm_config = handover_tdm_config()
	p1_tdm_idx = np.concatenate([np.arange(36),np.arange(-2,0)])
	p2_tdm_idx = np.concatenate([180+np.arange(36),np.arange(-2,0)])
	p1_vae_idx = np.arange(180)
	p2_vae_idx = np.arange(180) + 180

pred_mse_actions_ckpt = []
for a in test_dataset.actidx:
	pred_mse_actions_ckpt.append([])
	for i in range(a[0],a[1]):
		# x = test_data[i]
		x, label = test_dataset[i]
		x = torch.Tensor(x)
		label = torch.Tensor(label)
		x = torch.cat([x,label], dim=-1)
		seq_len, dims = x.shape
		x1_tdm = torch.Tensor(x[None,:,p1_tdm_idx]).to(device)
		x2_vae = torch.Tensor(x[None,:,p2_vae_idx]).to(device)
		
		# z1_d1_dist, d1_samples, d1_dist = tdm_1(x1_tdm, lens)
		d1_x1 = tdm_1.latent_mean(tdm_1.activation(tdm_1._encoder(x1_tdm)[0]))
		z2_d1 = tdm_2.output_mean(tdm_2._decoder(d1_x1))
		x2_tdm_out = vae._output(vae._decoder(z2_d1))

		pred_mse_actions_ckpt[-1] += ((x2_tdm_out - x2_vae)**2).reshape((seq_len, vae.window_size, vae.num_joints, vae.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()

s = ''
for mse in pred_mse_actions_ckpt:
	if len(mse) ==0:
		continue
	s += f'{np.mean(mse)*100:.3f} $\pm$ {np.std(mse)*100:.3f}\t'
print(s) # prints the Mean squared error and standard deviation for each interaction in the dataset
