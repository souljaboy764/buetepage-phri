import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import networks
from utils import *
from config import *

parser = argparse.ArgumentParser(description='Buetepage HRI Dynamics Training')
parser.add_argument('--human-ckpt', type=str, metavar='HUMAN-CKPT', default='logs/2023/hh_20hz_3joints_xvel/trial3/tdm/models/tdm_final.pth',
					help='Path to the Human dynamics checkpoint.')
parser.add_argument('--robot-ckpt', type=str, metavar='ROBOT-CKPT', default='logs/2023/pepper_20hz_3joints_xvel/trial3/models/final.pth',
					help='Path to the VAE checkpoint, where the TDM models will also be saved.')
parser.add_argument('--src', type=str, default='data/2023/pepper_20hz_3joints_xvel/tdm_data.npz', metavar='DATA',
					help='Path to read training and testing data (default: ./data/orig_hr/labelled_sequences.npz).')
args = parser.parse_args()
torch.manual_seed(128542)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(args.robot_ckpt)))
VAE_MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'dynamics', "models")
global_step = 0
if not os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(VAE_MODELS_FOLDER, 'final.pth')):
	print('Please use the same directory as the final VAE model')
	exit(-1)

if os.path.exists(os.path.join(MODELS_FOLDER,'hyperparams.npz')):
	hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	# hri_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	hri_config = hyperparams['hri_config'].item()
	config = hyperparams['global_config'].item()
else:
	hri_config = hri_config()
	config = global_config()

robot_vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
robot_vae_args = robot_vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
robot_vae_config = robot_vae_hyperparams['vae_config'].item()
robot_vae = networks.VAE(**(robot_vae_config.__dict__)).to(device)
ckpt = torch.load(args.robot_ckpt)
robot_vae.load_state_dict(ckpt['model'])
robot_vae.eval()

human_tdm_hyperparams = np.load(os.path.join(os.path.dirname(args.human_ckpt),'tdm_hyperparams.npz'), allow_pickle=True)
human_tdm_args = human_tdm_hyperparams['args'].item() # overwrite args if loading from checkpoint
human_tdm_config = human_tdm_hyperparams['tdm_config'].item()
human_tdm = networks.TDM(**(human_tdm_config.__dict__)).to(device)
ckpt = torch.load(args.human_ckpt)
human_tdm.load_state_dict(ckpt['model_1'])
human_tdm.eval()

hri = networks.HRIDynamics(**(hri_config.__dict__)).to(device)
ckpt = torch.load(os.path.join(MODELS_FOLDER, 'tdm_final.pth'))
hri.load_state_dict(ckpt['model'])
hri.eval()

print("Reading Data")
with np.load(args.src, allow_pickle=True) as data:
	test_data = [torch.Tensor(traj) for traj in data['test_data']]
	test_iterator = DataLoader(test_data, batch_size=1, shuffle=True)


# p1_tdm_idx = np.concatenate([np.arange(12),np.arange(-5,0)])
# r2_hri_idx = np.concatenate([480+np.arange(7),np.arange(-5,0)])
# r2_vae_idx = np.arange(280) + 480
p1_tdm_idx = np.concatenate([np.arange(18),np.arange(-5,0)])
r2_hri_idx = np.concatenate([90+np.arange(4),np.arange(-5,0)])
r2_vae_idx = 90 + np.arange(20)
actidx = np.array([[0,7],[7,15],[15,29],[29,39]])

print("Starting Evaluation")
mse_actions = []

print('Pred. MSE (all)\t\tPred. MSE w/o waving\t\tPred. MSE waving\t\tPred. MSE handshake\t\tPred. MSE rocket\t\tPred. MSE parachute')
print('mean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma')

for a in actidx:
	mse_actions.append([])
	for i in range(a[0],a[1]):
		x = test_data[i]
		seq_len, dims = x.shape
		x_p1_tdm = x[:,p1_tdm_idx].to(device)
		x_r2_gt = x[:,r2_vae_idx].to(device)
		x_r2_hri = x[:,r2_hri_idx].to(device)
		with torch.no_grad():
			_, _, d_x1_dist = human_tdm(x_p1_tdm, None)
			hri_input = torch.concat([x_r2_hri, d_x1_dist.mean], dim=-1)
			z_r2hri_dist, z_r2hri_samples = hri(hri_input, None)
			x_r2_gen = robot_vae._output(robot_vae._decoder(z_r2hri_dist.mean))

		mse_actions[-1] += ((x_r2_gt - x_r2_gen)**2).reshape((seq_len, robot_vae.window_size, robot_vae.num_joints, robot_vae.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()

pred_mse_ckpt = []
pred_mse_nowave_ckpt = []
for mse in mse_actions:
	pred_mse_ckpt+= mse
for mse in mse_actions[1:]:
	pred_mse_nowave_ckpt+= mse
s = f'{np.mean(pred_mse_ckpt):.4e}\t{np.std(pred_mse_ckpt):.4e}\t{np.mean(pred_mse_nowave_ckpt):.4e}\t{np.std(pred_mse_nowave_ckpt):.4e}'
for mse in mse_actions:
	s += f'\t{np.mean(mse):.4e}\t{np.std(mse):.4e}'
print(s)