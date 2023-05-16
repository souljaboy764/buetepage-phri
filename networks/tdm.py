import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from torch.nn.utils.rnn import pad_packed_sequence

class TDM(nn.Module):
	def __init__(self, **kwargs):
		super(TDM, self).__init__()
		for key in kwargs:
			setattr(self, key, kwargs[key])

		self.activation = getattr(nn, kwargs['activation'])()
		self.input_dim = self.num_joints * self.joint_dims + self.num_actions
		
		# enc_sizes = [self.input_dim] + self.encoder_sizes
		# self._encoder = nn.ModuleList([])
		# for i in range(len(enc_sizes)-1):
		# 	self._encoder.append(nn.LSTM(enc_sizes[i], enc_sizes[i+1]))

		encoder_layers = []
		for i in range(self.num_lstm_layers):
			encoder_layers.append(nn.LSTM(self.input_dim, self.lstm_hidden, 1, batch_first=True))
			encoder_layers.append(self.activation)
		self._encoder = nn.LSTM(self.input_dim, self.lstm_hidden, self.num_lstm_layers, batch_first=True)

		self.latent_mean = nn.Linear(self.lstm_hidden, self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.latent_std = nn.Sequential(nn.Linear(self.lstm_hidden, self.latent_dim), nn.Softplus())

		dec_sizes = [self.latent_dim] + self.decoder_sizes
		dec_layers = []
		for i in range(len(dec_sizes)-1):
			dec_layers.append(nn.Linear(dec_sizes[i-1], dec_sizes[i]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)

		self.output_mean = nn.Linear(dec_sizes[-1], self.output_dim)
		self.output_std = nn.Sequential(nn.Linear(dec_sizes[-1], self.output_dim), nn.Softplus())

	def forward(self, x, seq_len):
		enc,_ = self._encoder(x)
		enc, _ = pad_packed_sequence(enc, batch_first=True, total_length=seq_len)
		enc = self.activation(enc)

		d_dist = Normal(self.latent_mean(enc), self.latent_std(enc)+1e-4)
		if self.training:
			d_samples = d_dist.rsample()
		else:
			d_samples = d_dist.mean

		dec = self._decoder(d_samples)

		zd_dist = Normal(self.output_mean(dec), self.output_std(dec)+1e-4)

		return zd_dist, d_samples, d_dist
