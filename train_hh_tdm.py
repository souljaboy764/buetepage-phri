import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import networks
from config import global_config, human_tdm_config
from utils import *

p1_tdm_idx = np.concatenate([np.arange(12),np.arange(-5,0)])
p2_tdm_idx = np.concatenate([480+np.arange(12),np.arange(-5,0)])
p1_vae_idx = np.arange(480)
p2_vae_idx = np.arange(480) + 480
def run_iters_tdm(iterator, tdm, vae, optimizer):
	iters = 0
	total_jsd = []
	total_kl_1 = []
	total_kl_2 = []
	total_loss = []
	for i, (x, lens) in enumerate(iterator):
		if tdm.training:
			optimizer.zero_grad()
		batch_size, seq_len, dims = x.shape
		mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1) < lens.unsqueeze(1).repeat(1,seq_len)
		x1_tdm = x[:,:,p1_tdm_idx]
		x2_tdm = x[:,:,p2_tdm_idx]
		x1_vae = x[:,:,p1_vae_idx]
		x2_vae = x[:,:,p2_vae_idx]

		zd1_dist, d1_samples, d1_dist = tdm(torch.nn.utils.rnn.pack_padded_sequence(x1_tdm.to(device), lens, batch_first=True, enforce_sorted=False), seq_len)
		zd2_dist, d2_samples, d2_dist = tdm(torch.nn.utils.rnn.pack_padded_sequence(x2_tdm.to(device), lens, batch_first=True, enforce_sorted=False), seq_len)
		with torch.no_grad():
			zx1_dist = vae(x1_vae.to(device), True)
			zx2_dist = vae(x2_vae.to(device), True)

		kl_loss_1 = torch.distributions.kl_divergence(zd1_dist, zx1_dist)[mask].mean()
		kl_loss_2 = torch.distributions.kl_divergence(zd2_dist, zx2_dist)[mask].mean()
		
		d1_logprobs = d1_dist.log_prob(d1_samples)[mask]
		d2_logprobs = d2_dist.log_prob(d2_samples)[mask]
		jsd = JSD(d1_logprobs, d2_logprobs, log_targets=True, reduction='sum')

		loss = kl_loss_1 + kl_loss_2 + jsd
		
		total_jsd.append(jsd)
		total_kl_1.append(kl_loss_1)
		total_kl_2.append(kl_loss_2)
		total_loss.append(loss)

		if tdm.training:
			loss.backward()
			nn.utils.clip_grad_norm_(tdm.parameters(), 1.0)
			optimizer.step()
		iters += 1

	return total_loss, total_jsd, total_kl_1, total_kl_2, d1_samples[mask], d2_samples[mask], iters

def write_summaries_tdm(writer, loss, jsd, kl_1, kl_2, d1_samples, d2_samples, steps_done, prefix):
	writer.add_scalar(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/jsd_d', sum(jsd), steps_done)
	writer.add_scalar(prefix+'/kl_div_z1', sum(kl_1), steps_done)
	writer.add_scalar(prefix+'/kl_div_z2', sum(kl_2), steps_done)
	d = torch.concat([d1_samples[:100], d2_samples[:100]], dim=0)
	d_labels = torch.concat([torch.ones(100), torch.ones(100)+1],dim=0)

	writer.add_histogram('latents/d1_samples', d1_samples.mean(0), steps_done)
	writer.add_histogram('latents/d2_samples', d2_samples.mean(0), steps_done)

	# writer.add_embedding(d, metadata=d_labels, global_step=steps_done, tag=prefix+'/q(d|z)')

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--vae-ckpt', type=str, metavar='CKPT', default='logs/results/02231424/models/final.pth',
						help='Path to the VAE checkpoint, where the TDM models will also be saved.')
	parser.add_argument('--src', type=str, default='./data/orig_bothactors/tdm_data.npz', metavar='DATA',
						help='Path to read training and testin data (default: ./data/orig_bothactors/tdm_data.npz).')
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config = global_config()
	tdm_config = human_tdm_config()

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
		config = hyperparams['global_config'].item()
		tdm_config = hyperparams['tdm_config'].item()
	else:
		np.savez_compressed(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz'), args=args, global_config=config, tdm_config=tdm_config)

	vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	vae_args = vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	vae_config = vae_hyperparams['vae_config'].item()

	print("Creating Model and Optimizer")
	tdm = networks.TDM(**(tdm_config.__dict__)).to(device)
	optimizer = getattr(torch.optim, config.optimizer)(tdm.parameters(), lr=config.lr)

	if os.path.exists(os.path.join(MODELS_FOLDER, 'tdm_final.pth')):
		print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(MODELS_FOLDER, 'tdm_final.pth'))
		tdm.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		global_step = ckpt['epoch']

	vae = getattr(networks, vae_args.model)(**(vae_config.__dict__)).to(device)
	# ckpt = torch.load(os.path.join(VAE_MODELS_FOLDER, 'final.pth'))
	ckpt = torch.load(args.vae_ckpt)
	vae.load_state_dict(ckpt['model'])
	vae.eval()

	print("Reading Data")
	with np.load(args.src, allow_pickle=True) as data:
		train_data = [torch.Tensor(traj) for traj in data['train_data']]
		test_data = [torch.Tensor(traj) for traj in data['test_data']]

		while len(train_data)<tdm.batch_size:
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
	for k in tdm_config.__dict__:
		s += str(k) + ' : ' + str(tdm_config.__dict__[k]) + '\n'
	writer.add_text('human_tdm_config', s)

	writer.flush()

	print("Starting Epochs")
	for epoch in range(config.EPOCHS):
		tdm.train()
		packed_sequences = pack_padded_sequence(padded_sequences, lens, batch_first=True, enforce_sorted=False)
		train_loss, train_jsd, train_kl_1, train_kl_2, d1_samples, d2_samples, iters = run_iters_tdm(train_iterator, tdm, vae, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_tdm(writer, train_loss, train_jsd, train_kl_1, train_kl_2, d1_samples, d2_samples, steps_done, 'train')
		for name, param in tdm.named_parameters():
			if param.grad is None:
				continue
			value = param.reshape(-1)
			grad = param.grad.reshape(-1)
			# name=name.replace('.','/')
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		tdm.eval()
		with torch.no_grad():
			test_loss, test_jsd, test_kl_1, test_kl_2, d1_samples, d2_samples, iters = run_iters_tdm(test_iterator, tdm, vae, optimizer)
			write_summaries_tdm(writer, test_loss, test_jsd, test_kl_1, test_kl_2, d1_samples, d2_samples, steps_done, 'test')

		if epoch % config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, 'tdm_%0.4d.pth'%(epoch))
			torch.save({'model': tdm.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'tdm_final.pth')
	torch.save({'model': tdm.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
