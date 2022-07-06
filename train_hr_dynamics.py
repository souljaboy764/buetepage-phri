import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import networks
from config import *
from utils import *

p1_tdm_idx = np.concatenate([np.arange(12),np.arange(-5,0)])
r2_hri_idx = np.concatenate([480+np.arange(7),np.arange(-5,0)])
r2_vae_idx = np.arange(280) + 480
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_iters_hri(iterator, hri, robot_vae, human_tdm, optimizer):
	iters = 0
	total_loss = []
	for i, (x, lens) in enumerate(iterator):
		if hri.training:
			optimizer.zero_grad()
		batch_size, seq_len, dims = x.shape
		mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1) < lens.unsqueeze(1).repeat(1,seq_len)
		x_p1_tdm = x[:,:,p1_tdm_idx]
		x_r2_vae = x[:,:,r2_vae_idx]
		with torch.no_grad():
			_, _, d_x1_dist = human_tdm(torch.nn.utils.rnn.pack_padded_sequence(x_p1_tdm.to(device), lens, batch_first=True, enforce_sorted=False), seq_len)
			z_r2vae_dist = robot_vae(x_r2_vae.to(device), True)

		x_r2_hri = torch.concat([x[:,:,r2_hri_idx].to(device), d_x1_dist.mean], dim=-1)
		z_r2hri_dist, z_r2hri_samples = hri(torch.nn.utils.rnn.pack_padded_sequence(x_r2_hri.to(device), lens, batch_first=True, enforce_sorted=False), seq_len)
		
		loss = torch.distributions.kl_divergence(z_r2hri_dist, z_r2vae_dist)[mask].mean()
		total_loss.append(loss)

		if hri.training:
			loss.backward()
			# nn.utils.clip_grad_norm_(hri.parameters(), 1.0)
			optimizer.step()
		iters += 1

	return total_loss, z_r2hri_samples[mask], z_r2vae_dist.sample()[mask], iters

def write_summaries_hri(writer, loss, z_r2hri_samples, z_r2vae_samples, steps_done, prefix):
	writer.add_scalar(prefix+'/loss', sum(loss), steps_done)

	writer.add_histogram('latents/z_r2vae_samples', z_r2hri_samples.mean(0), steps_done)
	writer.add_histogram('latents/z_r2vae_samples', z_r2vae_samples.mean(0), steps_done)

	# writer.add_embedding(d, metadata=d_labels, global_step=steps_done, tag=prefix+'/q(d|z)')

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Training Human-Robot Interactive Dynamics')
	parser.add_argument('--human-ckpt', type=str, metavar='HUMAN-CKPT', default='logs/vae_hh_orig_oldcommit_AdamW_07011535/tdm/models/tdm_final.pth',
						help='Path to the Human dynamics checkpoint.')
	parser.add_argument('--robot-ckpt', type=str, metavar='ROBOT-CKPT', default='logs/vae_hr_AdamW_07031331/models/final.pth',
						help='Path to the VAE checkpoint, where the TDM models will also be saved.')
	parser.add_argument('--src', type=str, default='./data/orig_hr/tdm_data.npz', metavar='DATA',
						help='Path to read training and testing data (default: ./data/orig_hr/tdm_data.npz).')
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config = global_config()
	hri_config = hri_config()

	DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(args.robot_ckpt)))
	VAE_MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'dynamics', "models")
	os.makedirs(MODELS_FOLDER,exist_ok=True)
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'dynamics', "summary")
	global_step = 0
	if not os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(VAE_MODELS_FOLDER, 'final.pth')):
		print('Please use the same directory as the final VAE model')
		exit(-1)

	# if os.path.exists(os.path.join(MODELS_FOLDER,'hri_hyperparams.npz')):
	# 	hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	# 	args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	# 	config = hyperparams['global_config'].item()
	# 	tdm_config = hyperparams['tdm_config'].item()
	# else:
	np.savez_compressed(os.path.join(MODELS_FOLDER,'hri_hyperparams.npz'), args=args, global_config=config, hri_config=hri_config)

	robot_vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	robot_vae_args = robot_vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	robot_vae_config = robot_vae_hyperparams['vae_config'].item()
	robot_vae = getattr(networks, robot_vae_args.model)(**(robot_vae_config.__dict__)).to(device)
	ckpt = torch.load(args.robot_ckpt)
	robot_vae.load_state_dict(ckpt['model'])
	robot_vae.eval()

	human_tdm_hyperparams = np.load(os.path.join(os.path.dirname(args.human_ckpt),'tdm_hyperparams.npz'), allow_pickle=True)
	human_tdm_args = human_tdm_hyperparams['args'].item() # overwrite args if loading from checkpoint
	human_tdm_config = human_tdm_hyperparams['tdm_config'].item()
	human_tdm = networks.TDM(**(human_tdm_config.__dict__)).to(device)
	ckpt = torch.load(args.human_ckpt)
	human_tdm.load_state_dict(ckpt['model'])
	human_tdm.eval()

	print("Creating Model and Optimizer")
	hri = networks.HRIDynamics(**(hri_config.__dict__)).to(device)
	optimizer = getattr(torch.optim, config.optimizer)(hri.parameters(), lr=config.lr)

	# if os.path.exists(os.path.join(MODELS_FOLDER, 'tdm_final.pth')):
	# 	print("Loading Checkpoints")
	# 	ckpt = torch.load(os.path.join(MODELS_FOLDER, 'tdm_final.pth'))
	# 	tdm.load_state_dict(ckpt['model'])
	# 	optimizer.load_state_dict(ckpt['optimizer'])
	# 	global_step = ckpt['epoch']

	# ckpt = torch.load(os.path.join(VAE_MODELS_FOLDER, 'final.pth'))

	print("Reading Data")
	with np.load(args.src, allow_pickle=True) as data:
		train_data = [torch.Tensor(traj) for traj in data['train_data']]
		test_data = [torch.Tensor(traj) for traj in data['test_data']]

		while len(train_data)<hri.batch_size:
			train_data += train_data
			
		train_num = len(train_data)
		test_num = len(test_data)
		sequences = train_data + test_data
		lens = []
		for traj in sequences:
			lens.append(traj.shape[0])

		padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=1.)
		train_data = padded_sequences[:train_num]
		train_lens = lens[:train_num]
		test_data = padded_sequences[train_num:]
		test_lens = lens[train_num:]
		train_iterator = DataLoader(list(zip(train_data,train_lens)), batch_size=len(train_data), shuffle=True)
		test_iterator = DataLoader(list(zip(test_data,test_lens)), batch_size=len(test_data), shuffle=True)

	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	# tdm.eval()
	# writer.add_graph(model, torch.Tensor(test_data[:10]).to(device))
	# tdm.train()
	s = ''
	for k in config.__dict__:
		s += str(k) + ' : ' + str(config.__dict__[k]) + '\n'
	writer.add_text('global_config', s)

	s = ''
	for k in hri_config.__dict__:
		s += str(k) + ' : ' + str(hri_config.__dict__[k]) + '\n'
	writer.add_text('hri_config', s)

	writer.flush()

	print("Starting Epochs")
	for epoch in range(config.EPOCHS):
		hri.train()
		# packed_sequences = pack_padded_sequence(padded_sequences, lens, batch_first=True, enforce_sorted=False)
		train_loss, z_r2hri_samples, z_r2vae_samples, iters = run_iters_hri(train_iterator, hri, robot_vae, human_tdm, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_hri(writer, train_loss, z_r2hri_samples, z_r2vae_samples, steps_done, 'train')
		for name, param in hri.named_parameters():
			if param.grad is None:
				continue
			value = param.reshape(-1)
			grad = param.grad.reshape(-1)
			# name=name.replace('.','/')
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		hri.eval()
		with torch.no_grad():
			test_loss, z_r2hri_samples, z_r2vae_samples, iters = run_iters_hri(test_iterator, hri, robot_vae, human_tdm, optimizer)
			write_summaries_hri(writer, test_loss, z_r2hri_samples, z_r2vae_samples, steps_done, 'test')

		if epoch % config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, 'tdm_%0.4d.pth'%(epoch))
			torch.save({'model': hri.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'tdm_final.pth')
	torch.save({'model': hri.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
