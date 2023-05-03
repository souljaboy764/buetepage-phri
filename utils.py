import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample, affine_grid

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors_10 = get_cmap('tab10')
p1_tdm_idx = np.concatenate([np.arange(12),np.arange(-5,0)])
p2_tdm_idx = np.concatenate([480+np.arange(12),np.arange(-5,0)])
p1_vae_idx = np.arange(480)
p2_vae_idx = np.arange(480) + 480


def downsample_trajs(train_data, train_labels, downsample_len):
	theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(4,1,1)
	M = np.eye(downsample_len)*2
	for i in range(1,downsample_len):
		M[i][i-1] = M[i-1][i] = -1

	B = torch.Tensor(np.linalg.pinv(M) * 1e-6).to(device)
	L = torch.linalg.cholesky(B) # faster/memory-friendly to directly give cholesky
	num_trajs = len(train_data)
	n_augments = 30
	for i in range(num_trajs):
		train_data[i] = train_data[i].transpose(1,2,0) # 4, 3, seq_len
		train_data[i] = torch.Tensor(train_data[i]).to(device).unsqueeze(2) # 4, 3, 1 seq_len
		train_data[i] = torch.concat([train_data[i], torch.zeros_like(train_data[i])], dim=2) # 4, 3, 2 seq_len
		
		grid = affine_grid(theta, torch.Size([4, 3, 2, downsample_len]), align_corners=True)
		train_data[i] = grid_sample(train_data[i].type(torch.float32), grid, align_corners=True) # 4, 3, 2 downsample_len
		train_data[i] = train_data[i][:, :, 0].cpu().detach().numpy() # 4, 3, downsample_len
		train_data[i] = train_data[i].transpose(2,0,1) # downsample_len, 4, 3

		augments = torch.distributions.MultivariateNormal(torch.Tensor(train_data[i]).to(device).reshape(24, downsample_len), scale_tril=L).sample((n_augments,)).cpu().numpy()
		train_data = np.concatenate([train_data,augments.reshape(n_augments, downsample_len, 4, 6)], 0)
		train_labels = np.concatenate([train_labels,np.repeat(train_labels[i:i+1], augments.shape[0], axis=0)], 0)
	print('Augmented Sequences: Training', train_data.shape)
	print('Augmented Labels: Training', train_labels.shape)
	return train_data, train_labels

def MMD(x, y, reduction='mean'):
	"""Emprical maximum mean discrepancy with rbf kernel. The lower the result
	   the more evidence that distributions are the same.
	   https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html

	Args:
		x: first sample, distribution P
		y: second sample, distribution Q
		kernel: kernel type such as "multiscale" or "rbf"
	"""
	xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
	rx = (xx.diag().unsqueeze(0).expand_as(xx))
	ry = (yy.diag().unsqueeze(0).expand_as(yy))

	dxx = rx.t() + rx - 2. * xx # Used for A in (1)
	dyy = ry.t() + ry - 2. * yy # Used for B in (1)
	dxy = rx.t() + ry - 2. * zz # Used for C in (1)

	XX, YY, XY = (torch.zeros_like(xx),
				  torch.zeros_like(xx),
				  torch.zeros_like(xx))

	bandwidth_range = [10, 15, 20, 50]
	for a in bandwidth_range:
		XX += torch.exp(-0.5*dxx/a)
		YY += torch.exp(-0.5*dyy/a)
		XY += torch.exp(-0.5*dxy/a)

	if reduction=='none':
		return XX + YY - 2. * XY
	
	return getattr(torch, reduction)(XX + YY - 2. * XY)

def KLD(p, q, log_targets=False, reduction='sum'):
	if log_targets:
		kld = (p.exp()*(p - q))
	else:
		kld = (p*(p.log() - q.log()))
	
	if reduction is None:
		return kld
	return getattr(torch, reduction)(kld)

def JSD(pp, pq, qp, qq, log_targets=False, reduction='sum'):
	if log_targets:
		m_p = 0.5*(pp.exp() + pq.exp())
		m_q = 0.5*(qp.exp() + qq.exp())
		return 0.5*(KLD(pp.exp(), m_p, False, reduction) + KLD(qq.exp(), m_q, False, reduction))
	else:
		m_p = 0.5*(pp + pq)
		m_q = 0.5*(qp + qq)
		return 0.5*(KLD(pp, m_p, False, reduction) + KLD(qq, m_q, False, reduction))

def write_summaries_vae(writer, recon, kl, loss, x_gen, zx_samples, x, steps_done, prefix):
	writer.add_histogram(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)
	
	# # writer.add_embedding(zx_samples[:100],global_step=steps_done, tag=prefix+'/q(z|x)')
	# batch_size, window_size, num_joints, joint_dims = x_gen.shape
	# x_gen = x_gen[:5]
	# x = x[:5]
	
	# fig, ax = plt.subplots(nrows=5, ncols=num_joints, figsize=(28, 16), sharex=True, sharey=True)
	# fig.tight_layout(pad=0, h_pad=0, w_pad=0)

	# plt.subplots_adjust(
	# 	left=0.05,  # the left side of the subplots of the figure
	# 	right=0.95,  # the right side of the subplots of the figure
	# 	bottom=0.05,  # the bottom of the subplots of the figure
	# 	top=0.95,  # the top of the subplots of the figure
	# 	wspace=0.05,  # the amount of width reserved for blank space between subplots
	# 	hspace=0.05,  # the amount of height reserved for white space between subplots
	# )
	# x = x.cpu().detach().numpy()
	# x_gen = x_gen.cpu().detach().numpy()
	# for i in range(5):
	# 	for j in range(num_joints):
	# 		ax[i][j].set(xlim=(0, window_size - 1))
	# 		color_counter = 0
	# 		for dim in range(joint_dims):
	# 			ax[i][j].plot(x[i, :, j, dim], color=colors_10(color_counter%10))
	# 			ax[i][j].plot(x_gen[i, :, j, dim], linestyle='--', color=colors_10(color_counter % 10))
	# 			color_counter += 1

	# fig.canvas.draw()
	# writer.add_figure('sample reconstruction', fig, steps_done)
	# plt.close(fig)

def prepare_axis():
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	# plt.ion()
	ax.view_init(25, -155)
	ax.set_xlim3d([-0.05, 0.75])
	ax.set_ylim3d([-0.3, 0.5])
	ax.set_zlim3d([-0.8, 0.2])
	return fig, ax

def reset_axis(ax, variant = None, action = None, frame_idx = None):
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	zlim = ax.get_zlim()
	ax.cla()
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_facecolor('none')
	ax.set_xlim3d(xlim)
	ax.set_ylim3d(ylim)
	ax.set_zlim3d(zlim)
	title = ""
	if variant is not None and action is not None and frame_idx is not None:
		ax.set_title(variant + " " + action + "\nFrame: {}".format(frame_idx))
	return ax

def visualize_skeleton(ax, trajectory, **kwargs):
	# trajectory shape: W, J, D (window size x num joints x joint dims)
	# Assuming that num joints = 4 and dims = 3
	assert len(trajectory.shape) ==  3 and trajectory.shape[1] == 4 and trajectory.shape[2] == 3
	for w in range(trajectory.shape[0]):
		ax.plot(trajectory[w, :, 0], trajectory[w, :, 1], trajectory[w, :, 2], color='k', marker='o', **kwargs)
	
	return ax