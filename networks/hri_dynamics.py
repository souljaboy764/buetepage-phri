import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from torch.nn.utils.rnn import pad_packed_sequence

# HRI Dynamics Model for Human-Robot Interactions
class HRIDynamics(nn.Module):
	def __init__(self, **kwargs):
		super(HRIDynamics, self).__init__()
		for key in kwargs:
			setattr(self, key, kwargs[key])

		self.activation = getattr(nn, kwargs['activation'])()
		self.input_dim = self.num_joints * self.joint_dims + self.num_actions + self.human_dims
		
		# enc_sizes = [self.input_dim] + self.encoder_sizes
		# self._encoder = nn.ModuleList([])
		# for i in range(len(enc_sizes)-1):
		# 	self._encoder.append(nn.LSTM(enc_sizes[i], enc_sizes[i+1]))

		self._encoder_lstm = nn.LSTM(self.input_dim, self.lstm_hidden, self.num_lstm_layers, batch_first=True)
		self._encoder_linear = nn.Sequential(nn.Linear(self.lstm_hidden, self.linear_hidden),
										self.activation,
										nn.Linear(self.linear_hidden, self.linear_hidden),
										self.activation)

		self.latent_mean = nn.Linear(self.linear_hidden, self.latent_dim)
		# Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		self.latent_std = nn.Sequential(nn.Linear(self.linear_hidden, self.latent_dim), nn.Softplus())

	def forward(self, x, lstm_state):
		enc, lstm_state = self._encoder_lstm(x, lstm_state)
		# enc, _ = pad_packed_sequence(enc, batch_first=True, total_length=seq_len)
		enc = self._encoder_linear(enc)

		z_dist = Normal(self.latent_mean(enc), self.latent_std(enc)+1e-4)
		z_samples = z_dist.rsample()

		return z_dist, z_samples, lstm_state
