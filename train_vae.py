import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import networks
from config import global_config, human_vae_config, robot_vae_config
from utils import *

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
	parser = argparse.ArgumentParser(description='Buetepage et al. (2020) Training')
	parser.add_argument('--results', type=str, default='./logs/results/'+datetime.datetime.now().strftime("%m%d%H%M"), metavar='RES',
						help='Path for saving results (default: ./logs/results/MMDDHHmm).')
	parser.add_argument('--src', type=str, default='./data/orig/vae/data.npz', metavar='RES',
						help='Path to read training and testing data (default: ./data/orig/vae/data.npz).') # ./data/orig_hr/vae_data.npz for HRI
	parser.add_argument('--model', type=str, default='VAE', metavar='AE', choices=['AE', 'VAE', "AE_HRI", "VAE_HRI"],
						help='Which model to use (AE, VAE, AE_HRI, VAE_HRI) (default: VAE).')					
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	is_hri = args.model.split('_')[-1]=='HRI'
	args.model = args.model.split('_')[0]
	config = global_config()
	vae_config = robot_vae_config() if is_hri else human_vae_config()

	DEFAULT_RESULTS_FOLDER = args.results
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
	global_step = 0
	if not os.path.exists(DEFAULT_RESULTS_FOLDER):
		print("Creating Result Directories")
		os.makedirs(DEFAULT_RESULTS_FOLDER)
		os.makedirs(MODELS_FOLDER)
		os.makedirs(SUMMARIES_FOLDER)
		np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, global_config=config, vae_config=vae_config)

	elif os.path.exists(os.path.join(MODELS_FOLDER,'hyperparams.npz')):
		hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
		args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		config = hyperparams['global_config'].item()
		vae_config = hyperparams['vae_config'].item()

	print("Creating Model and Optimizer")
	model = getattr(networks, args.model)(**(vae_config.__dict__)).to(device)
	optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)

	if os.path.exists(os.path.join(MODELS_FOLDER, 'final.pth')):
		print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(MODELS_FOLDER, 'final.pth'))
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		global_step = ckpt['epoch']

	print("Reading Data")
	with np.load(args.src, allow_pickle=True) as data:
		train_data = np.array(data['train_data']).astype(np.float32)
		test_data = np.array(data['test_data']).astype(np.float32)
		if is_hri:
			# Training only the robot degrees of freedom
			train_data = train_data[:, -model.input_dim:]
			test_data = test_data[:, -model.input_dim:]
		else:
			num_samples, dim = train_data.shape
			train_p1 = train_data[:, :dim//2]
			train_p2 = train_data[:, dim//2:]
			test_p1 = test_data[:, :dim//2]
			test_p2 = test_data[:, dim//2:]
			train_data = np.vstack([train_p1, train_p2])
			test_data = np.vstack([test_p1, test_p2])

		train_iterator = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
		test_iterator = DataLoader(test_data, batch_size=model.batch_size, shuffle=True)

	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	
	s = ''
	for k in config.__dict__:
		s += str(k) + ' : ' + str(config.__dict__[k]) + '\n'
	writer.add_text('global_config', s)

	s = ''
	for k in vae_config.__dict__:
		s += str(k) + ' : ' + str(vae_config.__dict__[k]) + '\n'
	writer.add_text('human_vae_config', s)

	writer.flush()

	print("Starting Epochs")
	for epoch in range(config.EPOCHS):
		model.train()
		train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iters_vae(train_iterator, model, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_vae(writer, train_recon, train_kl, train_loss, x_gen, zx_samples, x, steps_done, 'train')
		params = []
		grads = []
		for name, param in model.named_parameters():
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		model.eval()
		with torch.no_grad():
			test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iters_vae(test_iterator, model, optimizer)
			write_summaries_vae(writer, test_recon, test_kl, test_loss, x_gen, zx_samples, x, steps_done, 'test')

		if epoch % config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
