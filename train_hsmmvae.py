import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import networks
from config import global_config, human_vae_config

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pbdlib as pbd
from utils import MMD

colors_10 = get_cmap('tab10')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def multi_variate_normal(x, mu, sigma=None, log=True, gmm=False, lmbda=None):
	"""
	Multivariatve normal distribution PDF

	:param x:		np.array([nb_samples, nb_dim])
	:param mu: 		np.array([nb_dim])
	:param sigma: 	np.array([nb_dim, nb_dim])
	:param log: 	bool
	:return:
	"""
	if not gmm:
		if type(sigma) is float:
			sigma = np.array(sigma, ndmin=2)
		if type(mu) is float:
			mu = np.array(mu, ndmin=1)
		if sigma is not None:
			sigma = sigma[None, None] if sigma.shape == () else sigma

		mu = mu[None] if mu.shape == () else mu
		x = x[:, None] if x.ndim == 1 else x

		dx = mu - x
		lmbda_ = np.linalg.inv(sigma) if lmbda is None else lmbda

		log_lik = -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', lmbda_, dx))

		if sigma is not None:
			log_lik -= 0.5 * (x.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1])
		else:
			log_lik -= 0.5 * (x.shape[1] * np.log(2 * np.pi) - np.linalg.slogdet(lmbda)[1])


		return log_lik if log else np.exp(log_lik)
	else:
		raise NotImplementedError


def forward_variable(hsmm, n_step):
	nbD = np.round(4 * n_step // hsmm.nb_states)
	hsmm.Pd = np.zeros((hsmm.nb_states, nbD))
	for i in range(hsmm.nb_states):
		hsmm.Pd[i, :] = multi_variate_normal(np.arange(nbD), hsmm.Mu_Pd[i], hsmm.Sigma_Pd[i]+1e-8, log=False)
		hsmm.Pd[i, :] = hsmm.Pd[i, :] / (np.sum(hsmm.Pd[i, :])+1e-8)

	h = np.zeros((hsmm.nb_states, n_step))

	ALPHA, S, h[:, 0] = hsmm._fwd_init_ts(nbD)

	for i in range(1, n_step):
		ALPHA, S, h[:, i] = hsmm._fwd_step_ts(ALPHA, S, nbD)

	h /= (np.sum(h, axis=0)+1e-8)
	return h


def run_iters_hsmmvae(iterator, hsmm, model, optimizer, equalize_classes = False):
	iters = 0
	total_recon = []
	total_reg = []
	total_loss = []
	# for i, x in enumerate(iterator.dataset):
	max_demos = np.diff(iterator.action_idx,axis=-1).max()
	label = 0
	z_dim = model.latent_dim
	
	# Learing Loop
	# for seg_idx in iterator.action_idx:
	actions = np.arange(len(iterator.action_idx))
	np.random.shuffle(actions)
	for a in actions:
		seg_idx = iterator.action_idx[a]
		label = a
		if equalize_classes:
			idx_list = np.concatenate([np.arange(seg_idx[0], seg_idx[1]), np.random.randint(seg_idx[0], seg_idx[1], max_demos + seg_idx[0] - seg_idx[1])])
		else:
			idx_list = np.arange(seg_idx[0], seg_idx[1])
		np.random.shuffle(idx_list)
		# nbD = np.round(4 * iterator.dataset[0].shape[0] // hsmm[label].nb_states)
		# hsmm[label].Pd = np.zeros((hsmm[label].nb_states, nbD))
		# for i in range(hsmm[label].nb_states):
		# 	hsmm[label].Pd[i, :] = multi_variate_normal(np.arange(nbD), hsmm[label].Mu_Pd[i], hsmm[label].Sigma_Pd[i], log=False)
		# 	hsmm[label].Pd[i, :] = hsmm[label].Pd[i, :] / np.sum(hsmm[label].Pd[i, :])

		# h = np.zeros((hsmm[label].nb_states, iterator.dataset[0].shape[0]))

		# ALPHA, S, h[:, 0] = hsmm[label]._fwd_init_ts(nbD)

		# for i in range(1, iterator.dataset[0].shape[0]):
		# 	ALPHA, S, h[:, i] = hsmm[label]._fwd_step_ts(ALPHA, S, nbD)

		# h /= np.sum(h, axis=0)

		alpha_hsmm, _, _, _, _ = hsmm[label].compute_messages(marginal=[], sample_size=iterator.dataset[0].shape[0])
		if np.any(np.isnan(alpha_hsmm)):
			print('OH NO!! NAN')
			alpha_hsmm = forward_variable(hsmm[label], iterator.dataset[0].shape[0])
		
		mu_prior = torch.Tensor([hsmm[label].mu[:,:z_dim], hsmm[label].mu[:,z_dim:]]).to(device)
		Sigma_prior = torch.Tensor([hsmm[label].sigma[:, :z_dim, :z_dim], hsmm[label].sigma[:, z_dim:, z_dim:]]).to(device)
		alpha_hsmm = torch.Tensor(alpha_hsmm).to(device)
		seq_alpha = alpha_hsmm.argmax(-1, keepdim=True).cpu().numpy()
		for i in idx_list:
			if model.training:
				optimizer.zero_grad()
			x = torch.Tensor(iterator.dataset[i]).to(device)
			seq_len, dims = x.shape
			x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
			x_gen, zpost_samples, zpost_dist = model(x)
			isnanx = torch.isnan(x_gen).any()
			isnanz = torch.isnan(zpost_samples).any()
			isnanz_mu = torch.isnan(zpost_dist.mean).any()
			if model.training:
				recon_loss = F.mse_loss(x[None].repeat(11,1,1,1), x_gen, reduction='sum')
			else:
				recon_loss = F.mse_loss(x, x_gen, reduction='sum')
			# reg_loss = model.latent_loss(zpost_samples, zpost_dist)
			
			# reg_loss = kl_divergence(zpost_dist, model.z_prior).mean()
			# reg_loss = MMD(zpost_samples, model.z_prior.sample(zpost_samples.shape[:-1]))

			# alpha_seq = np.argmax(alpha_hsmm, axis=0)
			# Upper bound for KL Div between 2 GMMs (here posterior is single Gaussian (GMM with 1 component))
			# Hershey, J.R. and Olsen, P.A., 2007, April. Approximating the Kullback Leibler divergence between Gaussian mixture models. In 2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07 (Vol. 4, pp. IV-317). IEEE.
			reg_loss = 0.
			z_dim = zpost_samples.shape[-1]
			for c in range(hsmm[label].nb_states):
				z_prior = torch.distributions.MultivariateNormal(mu_prior[:, c:c+1].repeat(1,seq_len,1), Sigma_prior[:, c:c+1].repeat(1,seq_len,1,1))
				kld = torch.distributions.kl_divergence(zpost_dist, z_prior).mean(0)
				reg_loss += (alpha_hsmm[c]*kld).mean()

			# z_prior = torch.distributions.MultivariateNormal(mu_prior[:, seq_alpha], Sigma_prior[:, seq_alpha])
			# reg_loss = torch.distributions.kl_divergence(zpost_dist, z_prior).mean()
						
			loss = recon_loss/model.beta + reg_loss

			total_recon.append(recon_loss)
			total_reg.append(reg_loss)
			total_loss.append(loss)

			if model.training:
				loss.backward()
				optimizer.step()
			iters += 1

	return total_recon, total_reg, total_loss, x_gen, zpost_samples, x, iters

def write_summaries_vae(writer, recon, kl, loss, x_gen, zx_samples, x, steps_done, prefix):
	writer.add_histogram(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)
	
	# writer.add_embedding(zx_samples[:100],global_step=steps_done, tag=prefix+'/q(z|x)')
	x_gen = x_gen[-1].reshape(-1, model.window_size, model.num_joints, model.joint_dims)
	x = x.reshape(-1, model.window_size, model.num_joints, model.joint_dims)
	batch_size, window_size, num_joints, joint_dims = x_gen.shape
	
	idx_list = np.random.randint(0,batch_size,5)
	x_gen = x_gen.detach().cpu().numpy()[idx_list]
	x = x.detach().cpu().numpy()[idx_list]
	
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
	parser.add_argument('--src', type=str, default='./data/ae_bip_downsample/vae_data.npz', metavar='RES',
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
		# train_lens = train_lens.astype(np.int32)
		# train_segs = np.cumsum(train_lens)
		# train_labels = np.zeros(149)
		# train_labels[24:54] = 1
		# train_labels[54:110] = 2
		# train_labels[110:] = 3
		train_actidx = np.array([[0,24],[24,54],[54,110],[110,149]])

		test_data, test_lens = np.array(data['test_data'])
		test_data = test_data.astype(np.float32)
		# test_lens = test_lens.astype(np.int32)
		# test_segs = np.cumsum(test_lens)
		# test_labels = np.zeros(39)
		# test_labels[7:15] = 1
		# test_labels[15:29] = 2
		# test_labels[29:] = 3
		test_actidx = np.array([[0,7],[7,15],[15,29],[29,39]])
		
		train_iterator = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
		# train_iterator.segments = train_segs
		train_iterator.action_idx = train_actidx
		# train_iterator.labels = train_labels
		train_iterator.seq_lens = train_lens
		test_iterator = DataLoader(test_data, batch_size=model.batch_size, shuffle=True)
		# test_iterator.segments = test_segs
		test_iterator.action_idx = test_actidx
		# test_iterator.labels = test_labels
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
	hsmm = []
	nb_dim = 2*model.latent_dim
	nb_states = args.hsmm_components
	for seg_idx in train_actidx:
		hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
		hsmm[-1].mu = np.zeros((nb_states, nb_dim))
		hsmm[-1].sigma = np.eye(nb_dim)[None].repeat(nb_states,0)
		hsmm[-1].Mu_Pd = np.zeros(nb_states)
		hsmm[-1].Sigma_Pd = np.ones(nb_states)
		hsmm[-1].Trans_Pd = np.ones((nb_states, nb_states))/nb_states

	for epoch in range(config.EPOCHS):
		model.train()
		train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iters_hsmmvae(train_iterator, hsmm, model, optimizer)
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
			test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iters_hsmmvae(test_iterator, hsmm, model, optimizer)
			write_summaries_vae(writer, test_recon, test_kl, test_loss, x_gen, zx_samples, x, steps_done, 'test')

			# Updating Prior
			for i in range(len(train_actidx)):
				s = train_actidx[i]
				x = torch.Tensor(train_iterator.dataset[s[0]:s[1]]).to(device)
				num_trajs, seq_len, dims = x.shape
				x = torch.concat([x[None, ..., :dims//2], x[None, ..., dims//2:]], dim=0) # x[0] = Agent 1, x[1] = Agent 2
				x_gen, zpost_samples, zpost_dist = model(x)
				z_encoded = torch.concat([zpost_dist.mean[0], zpost_dist.mean[1]], dim=-1).cpu().numpy() # (num_trajs, seq_len, 2*z_dim)
				hsmm[i].init_hmm_kbins(z_encoded)
				hsmm[i].em(z_encoded)

		if epoch % config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
