import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

from networks import AE

class VAE(AE):
	def __init__(self, **kwargs):
		super(VAE, self).__init__(**kwargs)
		
		self.post_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.post_std = nn.Sequential(nn.Linear(self.enc_sizes[-1], self.latent_dim), nn.Softplus())
		self.z_prior = Normal(self.z_prior_mean, self.z_prior_std)

	def forward(self, x, encode_only = False):
		enc = self._encoder(x)
		zpost_dist = Normal(self.post_mean(enc), self.post_std(enc))
		if encode_only:
			return zpost_dist
		zpost_samples = zpost_dist.rsample()
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, zpost_dist

	def latent_loss(self, zpost_samples, zpost_dist):
		return kl_divergence(zpost_dist, self.z_prior).mean()
