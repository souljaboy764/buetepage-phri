import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

from utils import MMD

class AE(nn.Module):
	def __init__(self, **kwargs):
		super(AE, self).__init__()
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

		self.latent = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		
		self.dec_sizes = [self.latent_dim] + self.hidden_sizes[::-1]
		dec_layers = []
		for i in range(len(self.dec_sizes)-1):
			dec_layers.append(nn.Linear(self.dec_sizes[i], self.dec_sizes[i+1]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)
		self._output = nn.Linear(self.dec_sizes[-1], self.input_dim) 

	def forward(self, x):
		enc = self._encoder(x)
		z_samples = self.latent(enc)
		x_gen = self._output(self._decoder(z_samples))
		return x_gen, z_samples, None
	
	def latent_loss(self, zpost_samples, zpost_dist):
		return 0

class VAE(AE):
	def __init__(self, **kwargs):
		super(VAE, self).__init__(**kwargs)
		
		self.latent_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.latent_std = nn.Sequential(nn.Linear(self.enc_sizes[-1], self.latent_dim), nn.Softplus())
		self.z_prior = Normal(self.z_prior_mean, self.z_prior_std)

	def forward(self, x):
		enc = self._encoder(x)
		zpost_dist = Normal(self.latent_mean(enc), self.latent_std(enc))
		zpost_samples = zpost_dist.rsample()
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, zpost_dist

	def latent_loss(self, zpost_samples, zpost_dist):
		return kl_divergence(zpost_dist, self.z_prior).sum()
		
class WAE(VAE):
	def __init__(self, **kwargs):
		super(WAE, self).__init__(**kwargs)

	def latent_loss(self, zpost_samples, zpost_dist):
		zprior_samples = self.z_prior.sample(zpost_samples.shape).to(zpost_samples.device)
		return MMD(zpost_samples, zprior_samples)

class TDM(nn.Module):
	def __init__(self, **kwargs):
		super(TDM, self).__init__()
		for key in kwargs:
			setattr(self, key, kwargs[key])

		self.activation = getattr(nn, kwargs['activation'])()
		self.input_dim = self.num_joints * self.joint_dims + self.num_actions
		
		enc_sizes = [self.input_dim] + self.encoder_sizes
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.LSTM(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self._encoder = nn.Sequential(*enc_layers)

		self.latent_mean = nn.Linear(enc_sizes[-1], self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.latent_std = nn.Sequential(nn.Linear(enc_sizes[-1], self.latent_dim), nn.Softplus())

		dec_sizes = [self.latent_dim] + self.decoder_sizes
		dec_layers = []
		for i in range(len(dec_sizes)-1):
			dec_layers.append(nn.Linear(dec_sizes[i-1], dec_sizes[i]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)

		self.output_mean = nn.Linear(dec_sizes[-1], self.output_dim)
		self.output_std = nn.Sequential(nn.Linear(dec_sizes[-1], self.output_dim), nn.Softplus())

	def forward(self, x):

		enc = self._encoder(x)

		d_dist = Normal(self.latent_mean(enc), self.latent_std(enc))
		d_samples = d_dist.rsample()

		dec = self._decoder(d_samples)

		zd_dist = Normal(self.output_mean(dec), self.output_std(dec))

		return zd_dist, d_samples, d_dist