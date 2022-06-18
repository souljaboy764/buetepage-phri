import torch
from torch import nn
from torch.nn import functional as F
from utils import MMD

from networks import VAE

class WAE(VAE):
	def __init__(self, **kwargs):
		super(WAE, self).__init__(**kwargs)

	def latent_loss(self, zpost_samples, zprior_samples):
		# zprior_samples = self.z_prior.sample(zpost_samples.shape).to(zpost_samples.device)
		return MMD(zpost_samples, zprior_samples)
