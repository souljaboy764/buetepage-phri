import torch
from torch import nn

from networks import AE

class VAE(AE):
	def __init__(self, **kwargs):
		super(VAE, self).__init__(**kwargs)
		
		self.post_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using logstd for now
		self.post_logstd = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		
	def forward(self, x, encode_only = False):
		enc = self._encoder(x)
		z_mean = self.post_mean(enc)
		if encode_only:
			return z_mean

		z_logstd = self.post_logstd(enc)
		z_std = z_logstd.exp()
			
		kld = 0.5*(z_std**2 + z_mean**2 - 1 - 2*z_logstd).sum(-1)
		if self.training:
			# Not mentioned how many samples used in paper so using one
			zpost_samples = z_mean + z_std*torch.randn_like(z_std)			
		else: 
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, z_mean, z_logstd, z_std, kld