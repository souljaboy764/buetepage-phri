import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import networks
from config import *
from utils import *

from phd_utils.dataloaders import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iters_vae(iterator, model, optimizer):
	iters = 0
	total_recon = []
	total_kl = []
	total_loss = []
	for i, x in enumerate(iterator):
		if model.training:
			optimizer.zero_grad()
		x = x.to(device)
		x_gen, zpost_samples, zpost_dist = model(x)

		recon_loss = F.mse_loss(x, x_gen, reduction='mean')
		kl_div = model.latent_loss(zpost_samples, zpost_dist)
		loss = recon_loss + model.beta*kl_div

		total_recon.append(recon_loss)
		total_kl.append(kl_div)
		total_loss.append(loss)

		if model.training:
			loss.backward()
			optimizer.step()
		iters += 1

	return total_recon, total_kl, total_loss, x_gen.reshape(-1, model.window_size, model.num_joints, model.joint_dims), zpost_samples, x.reshape(-1, model.window_size, model.num_joints, model.joint_dims), iters

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Buetepage et al. (2020) VAE Training')
	parser.add_argument('--results', type=str, default='./logs/debug', metavar='DST',
						help='Path for saving results (default: ./logs/debug).')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to load model weights. (default: None)')
	parser.add_argument('--model', type=str, default='HH', metavar='TYPE', choices=['HH', "PEPPER", 'NUISI_HH', "NUISI_PEPPER", "YUMI", 'ALAP'],
						help='Which model to use (HH, PEPPER, NUISI_HH, NUISI_PEPPER, YUMI or ALAP) (default: HH).')
	args = parser.parse_args()
	seed = np.random.randint(0,np.iinfo(np.int32).max)
	torch.manual_seed(seed)
	torch.autograd.set_detect_anomaly(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	is_hri = not (args.model == 'HH' or args.model == 'NUISI_HH' or args.model == 'ALAP')
	global_config = global_config()
	if args.model == 'HH' or args.model == 'NUISI_HH':
		config = human_vae_config()
	elif args.model == 'PEPPER' or args.model == 'NUISI_PEPPER':
		config = pepper_vae_config()
	elif args.model == 'YUMI':
		config = yumi_vae_config()
	elif args.model == 'ALAP':
		config = handover_vae_config()

	MODELS_FOLDER = os.path.join(args.results, "models")
	SUMMARIES_FOLDER = os.path.join(args.results, "summary")
	global_step = 0
	print("Creating Result Directories:",args.results)
	if not os.path.exists(args.results):
		os.makedirs(args.results)
	if not os.path.exists(MODELS_FOLDER):
		os.makedirs(MODELS_FOLDER)
	if not os.path.exists(SUMMARIES_FOLDER):
		os.makedirs(SUMMARIES_FOLDER)
	np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, global_config=global_config, config=config)

	print("Creating Model and Optimizer")
	model = networks.VAE(**(config.__dict__)).to(device)
	optimizer = getattr(torch.optim, global_config.optimizer)(model.parameters(), lr=global_config.lr)

	if args.ckpt is not None:
		ckpt = torch.load(args.ckpt)
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])

	print("Reading Data")
	if args.model =='HH':
		dataset = buetepage.HHWindowDataset
	elif args.model =='PEPPER':
		dataset = buetepage.PepperWindowDataset
	elif args.model =='NUISI_HH':
		dataset = nuisi.HHWindowDataset
	elif args.model =='NUISI_PEPPER':
		dataset = nuisi.PepperWindowDataset
	elif args.model =='YUMI':
		dataset = buetepage_hr.YumiWindowDataset
	elif args.model =='ALAP':
		dataset = alap.HHWindowDataset
	
	train_dataset = dataset(train=True, window_length=global_config.WINDOW_LEN, downsample=0.2)
	test_dataset = dataset(train=False, window_length=global_config.WINDOW_LEN, downsample=0.2)

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

	train_data = np.concatenate(train_dataset.traj_data).astype(np.float32)
	test_data = np.concatenate(test_dataset.traj_data).astype(np.float32)
	if args.model =='HH' or args.model == 'NUISI_HH':
		train_data = np.concatenate([train_data[:, :model.input_dim], train_data[:, model.input_dim:]])
		test_data = np.concatenate([test_data[:, :model.input_dim], test_data[:, model.input_dim:]])
	else:
		train_data = train_data[:, -model.input_dim:]
		test_data = test_data[:, -model.input_dim:]

	train_iterator = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
	test_iterator = DataLoader(test_data, batch_size=model.batch_size, shuffle=True)

	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	
	s = ''
	for k in global_config.__dict__:
		s += str(k) + ' : ' + str(global_config.__dict__[k]) + '\n'
	s += 'seed:'+str(seed)+'\n'
	writer.add_text('global_config', s)

	s = ''
	for k in config.__dict__:
		s += str(k) + ' : ' + str(config.__dict__[k]) + '\n'
	writer.add_text('config', s)

	writer.flush()

	print("Starting Epochs")
	for epoch in range(global_config.EPOCHS):
		model.train()
		train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iters_vae(train_iterator, model, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_vae(writer, train_recon, train_kl, train_loss, x_gen, zx_samples, x, steps_done, 'train_r' if is_hri else 'train')
		params = []
		grads = []
		for name, param in model.named_parameters():
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			model.eval()
			with torch.no_grad():
				test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iters_vae(test_iterator, model, optimizer)
				write_summaries_vae(writer, test_recon, test_kl, test_loss, x_gen, zx_samples, x, steps_done, 'test_r' if is_hri else 'test')

			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
