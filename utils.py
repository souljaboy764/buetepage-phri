import torch
import torch.nn.functional as F
import numpy as np

def MMD(x, y, reduction='mean'):
	"""Emprical maximum mean discrepancy with rbf kernel. The lower the result
	   the more evidence that distributions are the same.
	   https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html

	Args:
		x: first sample, distribution P
		y: second sample, distribution Q
		kernel: kernel type such as "multiscale" or "rbf"
	"""
	xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
	rx = (xx.diag().unsqueeze(0).expand_as(xx))
	ry = (yy.diag().unsqueeze(0).expand_as(yy))

	dxx = rx.t() + rx - 2. * xx # Used for A in (1)
	dyy = ry.t() + ry - 2. * yy # Used for B in (1)
	dxy = rx.t() + ry - 2. * zz # Used for C in (1)

	XX, YY, XY = (torch.zeros_like(xx),
				  torch.zeros_like(xx),
				  torch.zeros_like(xx))

	bandwidth_range = [10, 15, 20, 50]
	for a in bandwidth_range:
		XX += torch.exp(-0.5*dxx/a)
		YY += torch.exp(-0.5*dyy/a)
		XY += torch.exp(-0.5*dxy/a)

	if reduction=='none':
		return XX + YY - 2. * XY
	
	return getattr(torch, reduction)(XX + YY - 2. * XY)

def KLD(p, q, log_targets=False, reduction='sum'):
	if log_targets:
		kld = (p.exp()*(p - q))
	else:
		kld = (p*(p.log() - q.log()))
	
	if reduction is None:
		return kld
	return getattr(torch, reduction)(kld)

def JSD(p, q, log_targets=False, reduction='sum'):
	if log_targets:
		m = 0.5*(p.exp() + q.exp())
		return 0.5*(KLD(p.exp(), m, False, reduction) + KLD(q.exp(), m, False, reduction))
	else:
		m = 0.5*(p + q)
		return 0.5*(KLD(p, m, False, reduction) + KLD(q, m, False, reduction))
