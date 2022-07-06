import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import networks
from utils import *


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--vae-ckpt', type=str, metavar='CKPT', default='logs/vae_hh_orig_oldcommit_AdamW_07011535/models/final.pth',
						help='Path to the VAE checkpoint, where the TDM models will also be saved.')
	parser.add_argument('--src', type=str, default='./data/orig_bothactors/tdm_data.npz', metavar='DATA',
						help='Path to read training and testin data (default: ./data/orig_bothactors/tdm_data.npz).')
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(args.vae_ckpt)))
	VAE_MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'tdm', "models")
	os.makedirs(MODELS_FOLDER,exist_ok=True)
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
	global_step = 0
	if not os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(VAE_MODELS_FOLDER, 'final.pth')):
		print('Please use the same directory as the final VAE model')
		exit(-1)

	if os.path.exists(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz')):
		hyperparams = np.load(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz'), allow_pickle=True)
		args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		tdm_config = hyperparams['tdm_config'].item()
	else:
		print('No configs found')
		exit(-1)

	vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	vae_args = vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	vae_config = vae_hyperparams['vae_config'].item()

	print("Creating Model and Optimizer")
	tdm = networks.TDM(**(tdm_config.__dict__)).to(device)
	
	if os.path.exists(os.path.join(MODELS_FOLDER, 'tdm_final.pth')):
		print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(MODELS_FOLDER, 'tdm_final.pth'))
		tdm.load_state_dict(ckpt['model'])
		global_step = ckpt['epoch']
	else:
		print('No TDM model found')
		exit(-1)

	vae = getattr(networks, vae_args.model)(**(vae_config.__dict__)).to(device)
	ckpt = torch.load(args.vae_ckpt)
	vae.load_state_dict(ckpt['model'])
	vae.eval()

	print("Reading Data")
	with np.load(args.src, allow_pickle=True) as data:
		test_data_np = data['test_data']
		test_data = [torch.Tensor(traj) for traj in test_data_np]
		test_num = len(test_data)
		print(test_num,'Testing Trajecotries')
		lens = []
		for traj in test_data:
			lens.append(traj.shape[0])

		padded_sequences = pad_sequence(test_data, batch_first=True, padding_value=1.)
		test_iterator = DataLoader(list(zip(padded_sequences, lens)), batch_size=len(test_data), shuffle=False)
		p1_tdm_idx = np.concatenate([np.arange(12),np.arange(-5,0)])
		p2_tdm_idx = np.concatenate([480+np.arange(12),np.arange(-5,0)])
		p1_vae_idx = np.arange(480)
		p2_vae_idx = np.arange(480) + 480	

	print("Starting Evaluation")
	total_loss = []
	x_gen_tdm = []
	x_gen_autoencoding = []
	for i, (x, lens) in enumerate(test_iterator):
		batch_size, seq_len, dims = x.shape
		mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1) < lens.unsqueeze(1).repeat(1,seq_len)
		x1_tdm = x[:,:,p1_tdm_idx].to(device)
		x2_tdm = x[:,:,p2_tdm_idx].to(device)
		x1_vae = x[:,:,p1_vae_idx].to(device)
		x2_vae = x[:,:,p2_vae_idx].to(device)

		zd1_dist, d1_samples, d1_dist = tdm(torch.nn.utils.rnn.pack_padded_sequence(x1_tdm, lens, batch_first=True, enforce_sorted=False), seq_len)
		x_gen_i = vae._output(vae._decoder(zd1_dist.mean))
		
		loss = F.mse_loss(x2_vae, x_gen_i, reduction='none')[mask]
		print(loss.shape)
		total_loss.append(loss.detach().cpu().numpy())
		x_gen_tdm.append(x_gen_i.detach().cpu().numpy())

	total_loss = np.concatenate(total_loss,axis=0)
	# x_gen = np.concatenate(x_gen,axis=0)
	x_gen_tdm = np.array(x_gen_tdm)
	print(total_loss.mean())
	print(type(x_gen_tdm))
	print(type(test_data_np))
	np.savez_compressed('tdm_test.npz', x_gen=x_gen_tdm, test_data=test_data_np)