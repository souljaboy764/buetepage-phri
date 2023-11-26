import torch

import numpy as np
import os, argparse

import networks
from config import *
from utils import *

from phd_utils.dataloaders import *

parser = argparse.ArgumentParser(description='Buetepage HRI Dynamics Testing')
parser.add_argument('--robot-ckpt', type=str, metavar='ROBOT-CKPT', required=True,
					help='Path to the HRI Dynamics checkpoint.')
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HRI_MODELS_FOLDER = os.path.dirname(args.robot_ckpt)
if os.path.exists(os.path.join(HRI_MODELS_FOLDER,'hri_hyperparams.npz')):
	hyperparams = np.load(os.path.join(HRI_MODELS_FOLDER,'hri_hyperparams.npz'), allow_pickle=True)
	hri_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	hri_config = hyperparams['hri_config'].item()
	config = hyperparams['global_config'].item()
else:
	print(os.path.join(HRI_MODELS_FOLDER,'hyperparams.npz'),' does not exist')
	hri_config = hri_config()
	config = global_config()

robot_vae_hyperparams = np.load(os.path.join(os.path.dirname(hri_args.robot_ckpt),'hyperparams.npz'), allow_pickle=True)
robot_vae_args = robot_vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
robot_vae_config = robot_vae_hyperparams['config'].item()
robot_vae = networks.VAE(**(robot_vae_config.__dict__)).to(device)
ckpt = torch.load(hri_args.robot_ckpt)
robot_vae.load_state_dict(ckpt['model'])
robot_vae.eval()

human_tdm_hyperparams = np.load(os.path.join(os.path.dirname(hri_args.human_ckpt),'tdm_hyperparams.npz'), allow_pickle=True)
human_tdm_args = human_tdm_hyperparams['args'].item() # overwrite args if loading from checkpoint
human_tdm_config = human_tdm_hyperparams['tdm_config'].item()
human_tdm = networks.TDM(**(human_tdm_config.__dict__)).to(device)
ckpt = torch.load(hri_args.human_ckpt)
human_tdm.load_state_dict(ckpt['model_1'])
human_tdm.eval()

hri = networks.HRIDynamics(**(hri_config.__dict__)).to(device)
ckpt = torch.load(args.robot_ckpt)
hri.load_state_dict(ckpt['model'])
hri.eval()
p1_tdm_idx = np.concatenate([np.arange(18),np.arange(-4,0)])
p2_tdm_idx = np.concatenate([90+np.arange(18),np.arange(-4,0)])
p1_vae_idx = np.arange(90)
p2_vae_idx = np.arange(90) + 90
if robot_vae_args.model =='BP_PEPPER':
	dataset = buetepage.PepperWindowDataset
	hri_config.num_joints = 4
	r2_hri_idx = np.concatenate([90+np.arange(4),np.arange(-4,0)])
	r2_vae_idx = 90 + np.arange(20)
elif robot_vae_args.model =='NUISI_PEPPER':
	dataset = nuisi.PepperWindowDataset
	hri_config.num_joints = 4
	r2_hri_idx = np.concatenate([90+np.arange(4),np.arange(-4,0)])
	r2_vae_idx = 90 + np.arange(20)
elif robot_vae_args.model =='BP_YUMI':
	dataset = buetepage_hr.YumiWindowDataset
	hri_config.num_joints = 7
	r2_hri_idx = np.concatenate([90+np.arange(7),np.arange(-4,0)])
	r2_vae_idx = 90 + np.arange(35)

test_dataset = dataset(train=False, window_length=config.WINDOW_LEN, downsample=0.2)

test_dataset.labels = []
for idx in range(len(test_dataset.actidx)):
	for i in range(test_dataset.actidx[idx][0], test_dataset.actidx[idx][1]):
		label = np.zeros((test_dataset.traj_data[i].shape[0], len(test_dataset.actidx)))
		label[:, idx] = 1
		test_dataset.labels.append(label)


pred_mse_actions_ckpt = []

for a in test_dataset.actidx:
	pred_mse_actions_ckpt.append([])
	for i in range(a[0],a[1]):
		x,label = test_dataset[i]
		x = torch.Tensor(x)
		label = torch.Tensor(label)
		x = torch.cat([x,label], dim=-1)
		seq_len, dims = x.shape
		x_p1_tdm = x[:,p1_tdm_idx].to(device)
		x_r2_gt = x[:,r2_vae_idx].to(device)
		x_r2_hri = x[:,r2_hri_idx].to(device)
		tdm_lstm_state = None
		hri_lstm_state = None
		current_robot_state = x_r2_hri[0:1]
		x_r2_gen = []
		for t in range(seq_len):
			with torch.no_grad():
				_, _, d_x1_dist, tdm_lstm_state = human_tdm(x_p1_tdm[t:t+1], tdm_lstm_state)
				hri_input = torch.concat([current_robot_state, d_x1_dist.mean], dim=-1)
				z_r2hri_dist, z_r2hri_samples, hri_lstm_state = hri(hri_input, hri_lstm_state)
				robot_pred = robot_vae._output(robot_vae._decoder(z_r2hri_dist.mean))
				x_r2_gen.append(robot_pred)
				robot_pred = robot_pred.reshape((robot_vae.window_size, robot_vae.num_joints))
				current_robot_state[0, :robot_vae.num_joints] = robot_pred[0]
		x_r2_gen = torch.cat(x_r2_gen)
		pred_mse_actions_ckpt[-1] += ((x_r2_gt - x_r2_gen)**2).reshape((seq_len, robot_vae.window_size, robot_vae.num_joints, robot_vae.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()


s = ''
for mse in pred_mse_actions_ckpt:
	s += f'{np.mean(mse):.3f} $\pm$ {np.std(mse):.3f}\t'
print(s)
