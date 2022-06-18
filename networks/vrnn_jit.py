import torch
from torch import nn, jit
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from torch.nn.utils.rnn import *

from networks import WAE
from utils import MMD

class VRNNCell(jit.ScriptModule):
	def __init__(self, **kwargs):
		super(VRNNCell, self).__init__()
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
		
		self.post_mean = nn.Linear(self.enc_sizes[-1]+self.latent_dim, self.latent_dim)
		self.post_std = nn.Sequential(nn.Linear(self.enc_sizes[-1]+self.latent_dim, self.latent_dim), nn.Softplus())
		
		self.prior_mean = nn.Linear(self.latent_dim, self.latent_dim)
		self.prior_std = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.Softplus())
		
		self.rnn = nn.RNNCell(self.enc_sizes[-1] + self.latent_dim, self.latent_dim, nonlinearity='relu')
		
		self.dec_sizes = [2*self.latent_dim] + self.hidden_sizes[::-1]
		dec_layers = []
		for i in range(len(self.dec_sizes)-1):
			dec_layers.append(nn.Linear(self.dec_sizes[i], self.dec_sizes[i+1]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)
		self._output = nn.Linear(self.dec_sizes[-1], self.input_dim) 
	
	@jit.script_method
	def forward(self, x, hidden):
		enc = self._encoder(x)
		enc = self.activation(enc)
		xh = torch.cat([enc,hidden],dim=1)
		
		zpost_dist = (self.post_mean(xh), self.post_std(xh))
		zprior_dist = (self.prior_mean(hidden), self.prior_std(hidden))
		zpost_samples = zpost_dist[0] + zpost_dist[1]*torch.rand_like(zpost_dist[0])

		x_gen = self._output(self._decoder(torch.cat([zpost_samples,hidden],dim=1)))

		hidden = self.rnn(torch.cat([enc,zpost_samples], dim=1), hidden)
		return x_gen, zpost_samples, zpost_dist, zprior_dist, hidden

	def latent_loss(self, zpost_samples, zprior_samples):
		return MMD(zpost_samples, zprior_samples)

class VRNN(jit.ScriptModule):
	def __init__(self, **cell_args):
		super(VRNN, self).__init__()
		self.cell = VRNNCell(**cell_args)

	@jit.script_method
	def forward(self, input, state):
		# type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
		inputs = input.unbind(0)
		zpost_samples = torch.jit.annotate(List[Tensor], [])
		zprior_samples = torch.jit.annotate(List[Tensor], [])
		kl_loss = torch.jit.annotate(List[Tensor], [])
		gen = torch.zeros_like(input)
		for i in range(len(inputs)):
			gen[i], zpost_sample, zpost_dist, zprior_dist, state = self.cell(inputs[i], state)
			zpost_samples += [zpost_sample]
			zprior_samples += [zprior_dist[0] + zprior_dist[1]*torch.rand_like(zprior_dist[0])]
			kl_loss += [self.cell.latent_loss(zpost_samples[-1], zprior_samples[-1])]
		return gen, torch.stack(zpost_samples), torch.stack(zprior_samples), torch.stack(kl_loss), state
