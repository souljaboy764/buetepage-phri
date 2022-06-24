import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import networks
from config import global_config, human_vae_config

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pbdlib

colors_10 = get_cmap('tab10')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iters_vae(iterator, lens, model, optimizer):
	iters = 0
	total_recon = []
	total_kl = []
	total_loss = []
	for i, x in enumerate(iterator):
		if model.training:
			optimizer.zero_grad()
		x = x.to(device)
		x_gen, zpost_samples, zpost_dist = model(x)

		recon_loss = F.mse_loss(x, x_gen, reduction='sum')
		kl_div = model.latent_loss(zpost_samples, zpost_dist)
		loss = recon_loss/model.beta + kl_div

		total_recon.append(recon_loss)
		total_kl.append(kl_div) 
		total_loss.append(loss)

		if model.training:
			loss.backward()
			optimizer.step()
		iters += 1

	return total_recon, total_kl, total_loss, x_gen.reshape(-1, model.window_size, model.num_joints, model.joint_dims), zpost_samples, x.reshape(-1, model.window_size, model.num_joints, model.joint_dims), iters

def write_summaries_vae(writer, recon, kl, loss, x_gen, zx_samples, x, steps_done, prefix):
	writer.add_histogram(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)
	
	# writer.add_embedding(zx_samples[:100],global_step=steps_done, tag=prefix+'/q(z|x)')
	batch_size, window_size, num_joints, joint_dims = x_gen.shape
	x_gen = x_gen[:5]
	x = x[:5]
	
	fig, ax = plt.subplots(nrows=5, ncols=num_joints, figsize=(28, 16), sharex=True, sharey=True)
	fig.tight_layout(pad=0, h_pad=0, w_pad=0)

	plt.subplots_adjust(
		left=0.05,  # the left side of the subplots of the figure
		right=0.95,  # the right side of the subplots of the figure
		bottom=0.05,  # the bottom of the subplots of the figure
		top=0.95,  # the top of the subplots of the figure
		wspace=0.05,  # the amount of width reserved for blank space between subplots
		hspace=0.05,  # the amount of height reserved for white space between subplots
	)
	x = x.cpu().detach().numpy()
	x_gen = x_gen.cpu().detach().numpy()
	for i in range(5):
		for j in range(num_joints):
			ax[i][j].set(xlim=(0, window_size - 1))
			color_counter = 0
			for dim in range(joint_dims):
				ax[i][j].plot(x[i, :, j, dim], color=colors_10(color_counter%10))
				ax[i][j].plot(x_gen[i, :, j, dim], linestyle='--', color=colors_10(color_counter % 10))
				color_counter += 1

	fig.canvas.draw()
	writer.add_figure('sample reconstruction', fig, steps_done)
	plt.close(fig)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--results', type=str, default='./logs/results/'+datetime.datetime.now().strftime("%m%d%H%M"), metavar='RES',
						help='Path for saving results (default: ./logs/results/MMDDHHmm).')
	parser.add_argument('--src', type=str, default='./data/ae_bip_downsampled/vae_data.npz', metavar='RES',
						help='Path to read training and testin data (default: ./data/orig/vae/data.npz).')
	parser.add_argument('--prior', type=str, default='HSMM', metavar='P(Z)', choices=['None', 'RNN', 'BIP', 'HSMM'],
						help='Which prior to use for the VAE (default: None')	
	parser.add_argument('--hsmm-components', type=int, default=6, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 6).')						
	parser.add_argument('--model', type=str, default='VAE', metavar='ARCH', choices=['AE', 'VAE', 'WAE'],
						help='Path to read training and testin data (default: ./data/data/single_sample_per_action/data.npz).')					
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config = global_config()
	vae_config = human_vae_config()

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
		train_data, train_lens = np.array(data['train_data'])
		train_data = train_data.astype(np.float32)
		train_lens = train_lens.astype(np.int32)
		train_segs = np.cumsum(train_lens)
		train_actidx = np.array([[0,24],[24,54],[54,110],[110,149]])

		test_data, test_lens = np.array(data['test_data'])
		test_data = test_data.astype(np.float32)
		test_lens = test_lens.astype(np.int32)
		test_segs = np.cumsum(test_lens)
		test_actidx = np.array([[0,7],[7,15],[15,29],[29,39]])
		
		# train_data = np.array(data['train_data']).astype(np.float32)
		# test_data = np.array(data['test_data']).astype(np.float32)
		num_samples, dim = train_data.shape
		# train_p1 = train_data[:, :dim//2]
		# train_p2 = train_data[:, dim//2:]
		# test_p1 = test_data[:, :dim//2]
		# test_p2 = test_data[:, dim//2:]
		# train_data = np.vstack([train_p1, train_p2])
		# test_data = np.vstack([test_p1, test_p2])
		train_iterator = DataLoader(torch.Tensor(train_data).to(device), batch_size=model.batch_size, shuffle=True)
		train_iterator.segments = train_segs
		train_iterator.action_idx = train_actidx
		train_iterator.seq_lens = train_lens
		test_iterator = DataLoader(torch.Tensor(test_data).to(device), batch_size=model.batch_size, shuffle=True)
		test_iterator.segments = test_segs
		test_iterator.action_idx = test_actidx
		test_iterator.seq_lens = test_lens
	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	# model.eval()
	# writer.add_graph(model, torch.Tensor(test_data[:10]).to(device))
	# model.train()
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
