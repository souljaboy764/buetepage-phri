import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *

import numpy as np
import os, argparse

import networks
from utils import *

parser = argparse.ArgumentParser(description='SKID Training')
parser.add_argument('--vae-ckpt', type=str, metavar='CKPT', default='logs/092023/hh_20hz_3joints_xvel/',
					help='Path to the VAE checkpoint, where the TDM models will also be saved.')
parser.add_argument('--src', type=str, default='data/2023/hh_20hz_3joints_xvel/tdm_data.npz', metavar='DATA',
					help='Path to read training and testin data (default: ./data/orig_bothactors/tdm_data.npz).')
args = parser.parse_args()
torch.manual_seed(128542)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pred_mse = []
pred_mse_nowave = []
pred_mse_actions = [[],[],[],[]]
print('Trial\tPred. MSE (all)\t\tPred. MSE w/o waving\t\tPred. MSE waving\t\tPred. MSE handshake\t\tPred. MSE rocket\t\tPred. MSE parachute')
print('\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma')
for trial in range(4):
	vae_model_folder = os.path.join(args.vae_ckpt, f'trial{trial}', "models")
	vae_ckpt = os.path.join(vae_model_folder, 'final.pth')
	tdm_models_folder = os.path.join(args.vae_ckpt, f'trial{trial}', 'tdm', "models")
	if not os.path.exists(os.path.join(vae_model_folder, 'final.pth')):
		print('Please use the same directory as the final VAE model. Currently using',vae_model_folder)
		exit(-1)

	if os.path.exists(os.path.join(tdm_models_folder,'tdm_hyperparams.npz')):
		# print(os.path.join(tdm_models_folder,'tdm_hyperparams.npz'))
		hyperparams = np.load(os.path.join(tdm_models_folder,'tdm_hyperparams.npz'), allow_pickle=True)
		tdm_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		tdm_config = hyperparams['tdm_config'].item()
	else:
		print(os.path.join(tdm_models_folder,'tdm_hyperparams.npz'))
		print('No TDM configs found')
		exit(-1)

	vae_hyperparams = np.load(os.path.join(vae_model_folder,'hyperparams.npz'), allow_pickle=True)
	vae_args = vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	vae_config = vae_hyperparams['vae_config'].item()

	# print("Creating Model and Optimizer")
	tdm_1 = networks.TDM(**(tdm_config.__dict__)).to(device)
	tdm_2 = networks.TDM(**(tdm_config.__dict__)).to(device)

	if os.path.exists(os.path.join(tdm_models_folder, 'tdm_final.pth')):
		# print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(tdm_models_folder, 'tdm_final.pth'))
		tdm_1.load_state_dict(ckpt['model_1'])
		tdm_2.load_state_dict(ckpt['model_2'])
		global_step = ckpt['epoch']
	else:
		print('No TDM model found')
		exit(-1)

	vae = networks.VAE(**(vae_config.__dict__)).to(device)
	ckpt = torch.load(vae_ckpt)
	vae.load_state_dict(ckpt['model'])
	vae.eval()

	# print("Reading Data")
	# with np.load(args.src, allow_pickle=True) as data:
	# 	test_data_np = data['test_data']
	# 	test_data = [torch.Tensor(traj) for traj in test_data_np]

	from mild_hri.dataloaders import *
	dataset = nuisi.HHWindowDataset
	test_dataset = dataset(vae_args.src, train=False, window_length=vae_config.window_size, downsample=0.2)
	# print(len(test_dataset.traj_data), len(test_dataset.labels))
	# actidx = np.array([[0,7],[7,15],[15,29],[29,39]])

	# print("Starting Evaluation")
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

	pred_mse_ckpt = []
	pred_mse_nowave_ckpt = []
	for mse in pred_mse_actions_ckpt:
		pred_mse_ckpt+= mse
	for mse in pred_mse_actions_ckpt[1:]:
		pred_mse_nowave_ckpt+= mse
	s = f'{trial}'#\t{np.mean(pred_mse_ckpt):.4e}\t{np.std(pred_mse_ckpt):.4e}\t{np.mean(pred_mse_nowave_ckpt):.4e}\t{np.std(pred_mse_nowave_ckpt):.4e}'
	for mse in pred_mse_actions_ckpt:
		s += f'\t{np.mean(mse)*100:.3f} $\pm$ {np.std(mse)*100:.3f}'
	print(s)

	pred_mse += pred_mse_ckpt
	pred_mse_nowave += pred_mse_nowave_ckpt
	for i in range(4):
		pred_mse_actions[i] += pred_mse_actions_ckpt[i]

# s = f'all\t{np.mean(pred_mse):.4e}\t{np.std(pred_mse):.4e}\t{np.mean(pred_mse_nowave):.4e}\t{np.std(pred_mse_nowave):.4e}'
# for mse in pred_mse_actions:
# 	s += f'\t{np.mean(mse):.4e}\t{np.std(mse):.4e}'
# print(s)
np.savez_compressed('logs/mse/nuisihh_3joints_xvel.npz', np.array(pred_mse_actions, dtype=object))