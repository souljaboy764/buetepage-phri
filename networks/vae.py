import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

class VAE(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		for key in kwargs:
			setattr(self, key, kwargs[key])

		self.activation = getattr(nn, kwargs['activation'])()
		self.input_dim = self.num_joints * self.joint_dims * self.window_size
		
		self.enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(self.enc_sizes)-1):
			enc_layers.append(nn.Linear(self.enc_sizes[i], self.enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self._encoder = nn.Sequential(*enc_layers)
		
		self.dec_sizes = [self.latent_dim] + self.hidden_sizes[::-1]
		dec_layers = []
		for i in range(len(self.dec_sizes)-1):
			dec_layers.append(nn.Linear(self.dec_sizes[i], self.dec_sizes[i+1]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)
		self._output = nn.Linear(self.dec_sizes[-1], self.input_dim) 
		
		self.latent_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.latent_std = nn.Sequential(nn.Linear(self.enc_sizes[-1], self.latent_dim), nn.Softplus())
		self.z_prior = Normal(self.z_prior_mean, self.z_prior_std)

	def forward(self, x, encode_only = False):
		enc = self._encoder(x)
		zpost_dist = Normal(self.latent_mean(enc), self.latent_std(enc))
		if encode_only:
			return zpost_dist
		if not self.training:
			zpost_samples = zpost_dist.mean
		else:
			zpost_samples = zpost_dist.rsample()
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, zpost_dist

	def latent_loss(self, zpost_samples, zpost_dist):
		return kl_divergence(zpost_dist, self.z_prior).mean()
