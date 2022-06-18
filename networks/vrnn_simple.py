import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

from networks import AE
from utils import MMD

class VRNN_Simple(AE):
	def __init__(self, **kwargs):
		super(VRNN_Simple, self).__init__(**kwargs)
		
		self.post_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.post_std = nn.Sequential(nn.Linear(self.enc_sizes[-1], self.latent_dim), nn.Softplus())
		
		# self.z_prior = Normal(self.z_prior_mean, self.z_prior_std)
		
		self.prior_input_dim = self.num_joints * self.joint_dims * (self.window_size-1)

		self.prior_enc_sizes = [self.prior_input_dim] + self.hidden_sizes
		if self.reuse_encoder:
			self.prior_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
			self.prior_std = nn.Sequential(nn.Linear(self.enc_sizes[-1], self.latent_dim), nn.Softplus())
		else:
			prior_enc_layers = []
			for i in range(len(self.prior_enc_sizes)-1):
				prior_enc_layers.append(nn.Linear(self.prior_enc_sizes[i], self.prior_enc_sizes[i+1]))
				prior_enc_layers.append(self.activation)
			self._prior_encoder = nn.Sequential(*prior_enc_layers)
			self.prior_mean = nn.Linear(self.prior_enc_sizes[-1], self.latent_dim)
			self.prior_std = nn.Sequential(nn.Linear(self.prior_enc_sizes[-1], self.latent_dim), nn.Softplus())
		

	def forward(self, x, encode_only = False):
		enc = self._encoder(x)
		zpost_dist = Normal(self.post_mean(enc), self.post_std(enc))
		if encode_only:
			return zpost_dist
		if self.reuse_encoder:
			# enc0 = self._encoder[0].weight[:, self.prior_input_dim:] x[:, self.prior_input_dim:]
			enc0 = F.linear(x[:, :self.prior_input_dim], self._encoder[0].weight[:, :self.prior_input_dim], self._encoder[0].bias)
			prior_enc = self._encoder[1:](enc0)
			zprior_dist = Normal(self.prior_mean(prior_enc), self.prior_std(prior_enc))
		else:
			prior_enc = self._prior_encoder(x[:, :self.prior_input_dim])
			zprior_dist = Normal(self.prior_mean(prior_enc), self.prior_std(prior_enc))
		zpost_samples = zpost_dist.rsample()
		zprior_samples = zprior_dist.rsample()
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, zprior_samples

	def latent_loss(self, zpost_samples, zprior_samples):
		# return kl_divergence(zpost_dist, self.z_prior).mean()
		return MMD(zpost_samples, zprior_samples)
