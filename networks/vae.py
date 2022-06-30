import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl_divergence

from networks import AE


def batchNearestPD(A):
	"""Find the nearest positive-definite matrix to input
	A Python/Numpy port [1] of John D'Errico's `nearestSPD` MATLAB code [2], which
	credits [3].
	[1] https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
	[2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
	[3] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
	"""

	B = (A + torch.transpose(A,-2,-1))/2.
	_, s, V = torch.svd(B)
	sV = torch.bmm(torch.diag_embed(s),V)
	H = torch.bmm(torch.transpose(V,-2,-1),sV)

	A2 = (B + H) / 2

	A3 = (A2 + torch.transpose(A2,-2,-1)) / 2.

	if batchIsPD(A3):
		return A3

	spacing = torch.finfo(torch.float32).eps
	# The above is different from [1]. It appears that MATLAB's `chol` Cholesky
	# decomposition will accept matrixes with exactly 0-eigenvalue, whereas
	# Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
	# for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
	# will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
	# the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
	# `spacing` will, for Gaussian random matrixes of small dimension, be on
	# othe order of 1e-16. In practice, both ways converge, as the unit test
	# below suggests.
	with torch.no_grad():
		I = torch.eye(A.shape[-1]).repeat(A.shape[0],1,1).to(A.device)
	k = 1
	while not batchIsPD(A3):
		mineig = torch.real(torch.symeig(A3)[0].type(torch.cfloat))[:,0]
		v = (-mineig[:,None,None] * k**2 + spacing).repeat(1,A.shape[-1],A.shape[-1])
		A3 += I * v
		k += 1

	return A3

def batchIsPD(B):
	"""Returns true when input is positive-definite, via Cholesky"""
	try:
		torch.cholesky(B)
		return True
	except:
		return False


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
			z_mean_sampler = z_mean[None]
			z_std_sampler = z_std[None]
			zpost_samples = torch.concat([z_mean_sampler + z_std_sampler*torch.randn((10,)+z_std_sampler.shape), z_mean_sampler], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, z_mean, z_logstd, z_std, kld

class FullCovVAE(VAE):
	def __init__(self, **kwargs):
		super(VAE, self).__init__(**kwargs)
		
		self.post_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		self.post_cholesky = nn.Linear(self.enc_sizes[-1], (self.latent_dim*(self.latent_dim+1))//2)
		self.z_prior = Normal(self.z_prior_mean, self.z_prior_std)
		self.diag_idx = torch.arange(self.latent_dim)
		self.tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)

	def forward(self, x, encode_only = False, prior = None):
		enc = self._encoder(x)
		z_mean = self.post_mean(enc)
		
		if encode_only:
			return z_mean
		# Colesky Matrix Prediction 
		# Dorta et al. "Structured Uncertainty Prediction Networks" CVPR'18
		# Dorta et al. "Training VAEs Under Structured Residuals" 2018
		z_std = self.post_cholesky(enc)
		z_chol = torch.zeros(z_std.shape[:-1]+(self.latent_dim, self.latent_dim)).to(z_std.device)
		z_chol[..., self.tril_indices[0], self.tril_indices[1]] = z_std
		z_chol[..., self.diag_idx,self.diag_idx] = 2*torch.abs(z_chol[..., self.diag_idx,self.diag_idx]) + 1e-2

		# if prior is None:			
		# 	kld = 
		if self.training:
			z_mean_sampler = z_mean[None]
			z_std_sampler = z_std[None]
			zpost_samples = torch.concat([z_mean_sampler + z_std_sampler*torch.randn((10,)+z_std_sampler.shape), z_mean_sampler], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, (z_mean, z_chol, z_std)

	def latent_loss(self, zpost_samples, zpost_dist):
		if isinstance(self.z_prior, Normal):
			return kl_divergence(zpost_dist, self.z_prior).mean()
		if isinstance(self.z_prior, list):
			kld = 0
			for p in self.z_prior:
				kld += kl_divergence(zpost_dist, p).mean()
			return kld
