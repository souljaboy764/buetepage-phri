import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from torch.nn.utils.rnn import *

from networks import VAE

class VRNN(VAE):
	def __init__(self, **kwargs):
		super(VAE, self).__init__(**kwargs)
		
		self.post_mean = nn.Linear(self.enc_sizes[-1]+self.latent_dim, self.latent_dim)
		self.post_std = nn.Sequential(nn.Linear(self.enc_sizes[-1]+self.latent_dim, self.latent_dim), nn.Softplus())
		
		self.prior_mean = nn.Linear(self.latent_dim, self.latent_dim)
		self.prior_std = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.Softplus())
		
		self.rnn = nn.RNNCell(self.input_flat_size + self.z_dim, self.z_dim, nonlinearity='relu')

	def forward(self, x, hidden):
		enc, _ = self._encoder(x)
		enc = self.activation(enc)
		xh = torch.cat([enc,hidden],dim=1)
		
		zpost_dist = Normal(self.post_mean(xh), self.post_std(xh))
		zprior_dist = Normal(self.prior_mean(hidden), self.prior_std(hidden))
		zpost_samples = zpost_dist.rsample()

		zh = torch.cat([zpost_samples,hidden],dim=1)
		x_gen = self._output(self._decoder(zh))

		hidden = self.rnn(torch.cat(enc,zh), dim=1)
		return x_gen, zpost_samples, zpost_dist

	def latent_loss(self, zpost_samples, zpost_dist):
		return kl_divergence(zpost_dist, self.z_prior).mean()
