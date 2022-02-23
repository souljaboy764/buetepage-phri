import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

class VAE(nn.Module):
	def __init__(self, **kwargs):
		super(VAE, self).__init__()
		for key in kwargs:
			setattr(self, key, kwargs[key])

		self.activation = getattr(nn, kwargs['activation'])()
		self.input_dim = self.num_joints * self.joint_dims * self.window_size
		
		enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(enc_sizes)-1):
			enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self._encoder = nn.Sequential(*enc_layers)

		self.latent_mean = nn.Linear(enc_sizes[-1], self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.latent_std = nn.Sequential(nn.Linear(enc_sizes[-1], self.latent_dim), nn.Softplus())

		dec_sizes = [self.latent_dim] + self.hidden_sizes[::-1]
		dec_layers = []
		for i in range(len(dec_sizes)-1):
			dec_layers.append(nn.Linear(dec_sizes[i], dec_sizes[i+1]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)
		self._output = nn.Linear(dec_sizes[-1], self.input_dim) 

		# self.z_prior = torch.distributions.Normal(self.z_prior_mean, self.z_prior_std)

	def forward(self, x):

		enc = self._encoder(x)

		zx_mean = self.latent_mean(enc)
		zx_std = self.latent_std(enc)
		zx_samples = zx_mean + torch.rand_like(zx_mean) * zx_std
		zprior_samples = self.z_prior_mean + torch.rand_like(zx_mean) * self.z_prior_std
		x_gen = self._output(self._decoder(zx_samples))

		return x_gen, zx_samples, zx_mean, zx_std, zprior_samples

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