import torch

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# p1_tdm_idx = np.concatenate([np.arange(12),np.arange(-5,0)])
# p2_tdm_idx = np.concatenate([480+np.arange(12),np.arange(-5,0)])
# p1_vae_idx = np.arange(480)
# p2_vae_idx = np.arange(480) + 480

# p1_tdm_idx = np.concatenate([np.arange(18),np.arange(-4,0)])
# p2_tdm_idx = np.concatenate([90+np.arange(18),np.arange(-4,0)])
# p1_vae_idx = np.arange(90)
# p2_vae_idx = np.arange(90) + 90

p1_tdm_idx = np.concatenate([np.arange(36),np.arange(-2,0)])
p2_tdm_idx = np.concatenate([180+np.arange(36),np.arange(-2,0)])
p1_vae_idx = np.arange(180)
p2_vae_idx = np.arange(180) + 180

r2_hri_idx = np.concatenate([90+np.arange(7),np.arange(-4,0)])
r2_vae_idx = 90 + np.arange(35)
# r2_hri_idx = np.concatenate([90+np.arange(4),np.arange(-4,0)])
# r2_vae_idx = 90 + np.arange(20)

def KLD(p, q, log_targets=False, reduction='sum'):
	if log_targets:
		kld = (p.exp()*(p - q))
	else:
		kld = (p*(p.log() - q.log()))
	
	if reduction is None:
		return kld
	return getattr(torch, reduction)(kld)

def JSD(pp, pq, qp, qq, log_targets=False, reduction='sum'):
	if log_targets:
		m_p = 0.5*(pp.exp() + pq.exp())
		m_q = 0.5*(qp.exp() + qq.exp())
		return 0.5*(KLD(pp.exp(), m_p, False, reduction) + KLD(qq.exp(), m_q, False, reduction))
	else:
		m_p = 0.5*(pp + pq)
		m_q = 0.5*(qp + qq)
		return 0.5*(KLD(pp, m_p, False, reduction) + KLD(qq, m_q, False, reduction))

def write_summaries_vae(writer, recon, kl, loss, x_gen, zx_samples, x, steps_done, prefix):
	writer.add_histogram(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)

def prepare_axis_plotly():
	fig = make_subplots(rows=1, cols=1,
					specs=[[{'is_3d': True}]],
					print_grid=False)
	# fig.view_init(25, -155)
	fig.update_layout(
	    scene = dict(
			xaxis = dict(range=[-0.05, 0.75],),
			yaxis = dict(range=[-0.3, 0.5],),
			zaxis = dict(range=[-0.8, 0.2],),
		)
	)
	return fig

def plotly_skeleton(fig, trajectory, update=False, **kwargs):
	# trajectory shape: W, J, D (window size x num joints x joint dims)
	for w in range(trajectory.shape[0]):
		if update:
			fig.update_traces(patch={'x':trajectory[w, :, 0], 'y':trajectory[w, :, 1], 'z':trajectory[w, :, 2]}, selector=kwargs['start_idx']+w)
		else:
			fig.add_trace(go.Scatter3d(
				x=trajectory[w, :, 0],
				y=trajectory[w, :, 1],
				z=trajectory[w, :, 2],
				mode='markers',
				marker=dict(
					size=10,
					color=kwargs['color'],
					opacity=(w+1)/trajectory.shape[0]
				),
				line=dict(
					color='black',
					width=4,
					dash=kwargs['dash']
				)

			))
		
	return fig