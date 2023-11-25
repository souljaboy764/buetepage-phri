import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, argparse

import networks
from config import *
from utils import *

from phd_utils.dataloaders import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iters_tdm(iterator, tdm, vae, optimizer):
	tdm_1, tdm_2 = tdm
	iters = 0
	total_jsd = []
	total_kl_1 = []
	total_kl_2 = []
	total_loss = []
	for i, (x, label) in enumerate(iterator):
		if tdm_1.training:
			optimizer.zero_grad()

		x = torch.Tensor(x[0])
		label = torch.Tensor(label[0])
		x = torch.cat([x,label], dim=-1)
		x1_tdm = x[:,p1_tdm_idx]
		x2_tdm = x[:,p2_tdm_idx]
		x1_vae = x[:,p1_vae_idx]
		x2_vae = x[:,p2_vae_idx]

		zd1_dist, d1_samples, d1_dist, _ = tdm_1(x1_tdm.to(device), None)
		zd2_dist, d2_samples, d2_dist, _ = tdm_2(x2_tdm.to(device), None)
		with torch.no_grad():
			zx1_dist = vae(x1_vae.to(device), True)
			zx2_dist = vae(x2_vae.to(device), True)

		kl_loss_1 = torch.distributions.kl_divergence(zd1_dist, zx1_dist).mean()
		kl_loss_2 = torch.distributions.kl_divergence(zd2_dist, zx2_dist).mean()

		d11_logprobs = d1_dist.log_prob(d1_samples)
		d12_logprobs = d2_dist.log_prob(d1_samples)
		d21_logprobs = d1_dist.log_prob(d2_samples)
		d22_logprobs = d2_dist.log_prob(d2_samples)
		jsd = JSD(d11_logprobs, d12_logprobs, d21_logprobs, d22_logprobs, log_targets=True, reduction='mean')

		loss = kl_loss_1 + kl_loss_2 + jsd
		
		total_jsd.append(jsd)
		total_kl_1.append(kl_loss_1)
		total_kl_2.append(kl_loss_2)
		total_loss.append(loss)

		if tdm_1.training:
			loss.backward()
			optimizer.step()
		iters += 1

	return total_loss, total_jsd, total_kl_1, total_kl_2, d1_samples, d2_samples, iters

def write_summaries_tdm(writer, loss, jsd, kl_1, kl_2, d1_samples, d2_samples, steps_done, prefix):
	writer.add_scalar(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/jsd_d', sum(jsd), steps_done)
	writer.add_scalar(prefix+'/kl_div_z1', sum(kl_1), steps_done)
	writer.add_scalar(prefix+'/kl_div_z2', sum(kl_2), steps_done)

	writer.add_histogram('latents/d1_samples', d1_samples.mean(0), steps_done)
	writer.add_histogram('latents/d2_samples', d2_samples.mean(0), steps_done)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Buetepage et al. (2020) TDM Training')
	parser.add_argument('--vae-ckpt', type=str, metavar='CKPT', required=True,
						help='Path to the VAE checkpoint, where the TDM models will also be saved.')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to load model weights. (default: None)')
	args = parser.parse_args()
	torch.manual_seed(42) # answer to life universe and everything OP
	torch.autograd.set_detect_anomaly(True)

	config = global_config()

	DEFAULT_RESULTS_FOLDER = os.path.dirname(os.path.dirname(args.vae_ckpt))
	VAE_MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'tdm', "models")
	os.makedirs(MODELS_FOLDER,exist_ok=True)
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
	global_step = 0
	if not os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(VAE_MODELS_FOLDER, 'final.pth')):
		print('Please use the same directory as the final VAE model')
		exit(-1)

	vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	vae_args = vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	vae_config = vae_hyperparams['config'].item()

	if vae_args.model == 'BP_HH' or vae_args.model == 'NUISI_HH':
		tdm_config = human_tdm_config()
	elif vae_args.model == 'ALAP':
		tdm_config = handover_tdm_config()

	np.savez_compressed(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz'), args=args, global_config=config, tdm_config=tdm_config)

	print("Creating Model and Optimizer")
	tdm_1 = networks.TDM(**(tdm_config.__dict__)).to(device)
	tdm_2 = networks.TDM(**(tdm_config.__dict__)).to(device)
	optimizer = getattr(torch.optim, config.optimizer)(list(tdm_1.parameters()) + list(tdm_2.parameters()), lr=config.lr)

	# if os.path.exists(os.path.join(MODELS_FOLDER, 'tdm_final.pth')):
	if args.ckpt is not None:
		print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(MODELS_FOLDER, 'tdm_final.pth'))
		tdm_1.load_state_dict(ckpt['model_1'])
		tdm_2.load_state_dict(ckpt['model_2'])
		optimizer.load_state_dict(ckpt['optimizer'])
		global_step = ckpt['epoch']

	vae = networks.VAE(**(vae_config.__dict__)).to(device)
	ckpt = torch.load(args.vae_ckpt)
	vae.load_state_dict(ckpt['model'])
	vae.eval()

	print("Reading Data")
	if vae_args.model =='BP_HH':
		dataset = buetepage.HHWindowDataset
	elif vae_args.model =='NUISI_HH':
		dataset = nuisi.HHWindowDataset
	elif vae_args.model =='ALAP':
		dataset = alap.HHWindowDataset

	train_dataset = dataset(train=True, window_length=config.WINDOW_LEN, downsample=0.2)
	test_dataset = dataset(train=False, window_length=config.WINDOW_LEN, downsample=0.2)
	
	train_dataset.labels = []
	for idx in range(len(train_dataset.actidx)):
		for i in range(train_dataset.actidx[idx][0], train_dataset.actidx[idx][1]):
			label = np.zeros((train_dataset.traj_data[i].shape[0], len(train_dataset.actidx)))
			label[:, idx] = 1
			train_dataset.labels.append(label)

	test_dataset.labels = []
	for idx in range(len(test_dataset.actidx)):
		for i in range(test_dataset.actidx[idx][0], test_dataset.actidx[idx][1]):
			label = np.zeros((test_dataset.traj_data[i].shape[0], len(test_dataset.actidx)))
			label[:, idx] = 1
			test_dataset.labels.append(label)

	train_iterator = DataLoader(train_dataset, batch_size=1, shuffle=True)
	test_iterator = DataLoader(test_dataset, batch_size=1, shuffle=True)

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

	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
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
		tdm_1.train()
		tdm_2.train()
		train_loss, train_jsd, train_kl_1, train_kl_2, d1_samples, d2_samples, iters = run_iters_tdm(train_iterator, [tdm_1, tdm_2], vae, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_tdm(writer, train_loss, train_jsd, train_kl_1, train_kl_2, d1_samples, d2_samples, steps_done, 'train')
		for name, param in list(tdm_1.named_parameters())+list(tdm_2.named_parameters()):
			if param.grad is None:
				continue
			value = param.reshape(-1)
			grad = param.grad.reshape(-1)
			# name=name.replace('.','/')
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		tdm_1.eval()
		tdm_2.eval()
		with torch.no_grad():
			test_loss, test_jsd, test_kl_1, test_kl_2, d1_samples, d2_samples, iters = run_iters_tdm(test_iterator, [tdm_1, tdm_2], vae, optimizer)
			write_summaries_tdm(writer, test_loss, test_jsd, test_kl_1, test_kl_2, d1_samples, d2_samples, steps_done, 'test')

		if epoch % config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, 'tdm_%0.4d.pth'%(epoch))
			torch.save({'model_1': tdm_1.state_dict(), 'model_2': tdm_2.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'tdm_final.pth')
	torch.save({'model_1': tdm_1.state_dict(), 'model_2': tdm_2.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
