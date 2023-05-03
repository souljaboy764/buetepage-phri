import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *

import numpy as np
import os, argparse

import networks
from utils import *

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--vae-ckpt', type=str, metavar='CKPT', default='logs/vae_hh_orig_oldcommit_AdamW_07011535_tdmfixed/models/final.pth',
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
	if not os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(VAE_MODELS_FOLDER, 'final.pth')):
		print('Please use the same directory as the final VAE model')
		exit(-1)

	if os.path.exists(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz')):
		hyperparams = np.load(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz'), allow_pickle=True)
		args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		tdm_config = hyperparams['tdm_config'].item()
	else:
		print('No TDM configs found')
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
	total_loss_1 = []
	total_loss_2 = []
	x1_vae_gen = []
	x2_vae_gen = []
	x_tdm_gen = []
	for i, (x, lens) in enumerate(test_iterator):
		batch_size, seq_len, dims = x.shape
		mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1) < lens.unsqueeze(1).repeat(1,seq_len)
		x1_tdm = x[:,:,p1_tdm_idx].to(device)
		# x2_tdm = x[:,:,p2_tdm_idx].to(device)
		x1_vae = x[:,:,p1_vae_idx].to(device)
		x2_vae = x[:,:,p2_vae_idx].to(device)

		
		x1_vae_out, _, _ = vae(x1_vae)
		x2_vae_out, _, _ = vae(x2_vae)

		# didn't fully understand how p(d|h_1) can be used to generate p(z_1|d) and p(z_2|d) 
		# since the generated z_1 and z_2 (and x_1 and x_2) would belong to the same general distribution obtained from p(d|h_1)
		#
		# Is it the case that they have two separate AEs for each actor even in the HH case?
		z_d1_dist, d1_samples, d1_dist = tdm(torch.nn.utils.rnn.pack_padded_sequence(x1_tdm, lens, batch_first=True, enforce_sorted=False), seq_len)
		x_tdm_out = vae._output(vae._decoder(z_d1_dist.mean))
		
		loss_1 = F.mse_loss(x1_vae, x_tdm_out, reduction='none')[mask]
		loss_2 = F.mse_loss(x2_vae, x_tdm_out, reduction='none')[mask]
		
		x1_vae_gen.append(x1_vae_out.detach().cpu().numpy())
		x2_vae_gen.append(x2_vae_out.detach().cpu().numpy())
		x_tdm_gen.append(x_tdm_out.detach().cpu().numpy())
		total_loss_1.append(loss_1.detach().cpu().numpy())
		total_loss_2.append(loss_2.detach().cpu().numpy())

	total_loss_1 = np.concatenate(total_loss_1,axis=0).reshape((-1,40,4,3)).sum(-1).mean(-1)#.mean(-1)
	total_loss_2 = np.concatenate(total_loss_2,axis=0).reshape((-1,40,4,3)).sum(-1).mean(-1)#.mean(-1)
	x_tdm_gen = np.array(x_tdm_gen)
	x1_vae_gen = np.array(x1_vae_gen)
	x2_vae_gen = np.array(x2_vae_gen)
	print(total_loss_1.shape, total_loss_2.shape)
	np.savez_compressed('recon_error_bp.npz', error=total_loss_2)
	# np.savez_compressed('tdm_test.npz', x_tdm_gen=x_tdm_gen, x1_vae_gen=x1_vae_gen, x2_vae_gen=x2_vae_gen, test_data=test_data_np)