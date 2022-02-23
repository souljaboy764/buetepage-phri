import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

colors_10 = get_cmap('tab10')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iters(iterator, model, optimizer):
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

def write_summaries(writer, recon, kl, loss, x_gen, zx_samples, x, steps_done, prefix):
	writer.add_histogram(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)
	
	writer.add_embedding(zx_samples[:100],global_step=steps_done, tag=prefix+'/q(z|x)')
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

def MMD(x, y):
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

	return torch.mean(XX + YY - 2. * XY)
