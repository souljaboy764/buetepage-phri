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
parser.add_argument('--robot-ckpt', type=str, metavar='ROBOT-CKPT', default='logs/092023/nuisipepper_3joints_xvel/',
					help='Path to the VAE checkpoint, where the TDM models will also be saved.')
args = parser.parse_args()
torch.manual_seed(128542)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_mse = []
pred_mse_nowave = []
pred_mse_actions = [[],[],[],[]]
best_action_mse = [1000,1000,1000,1000]
print('Pred. MSE waving\t\tPred. MSE handshake\t\tPred. MSE rocket\t\tPred. MSE parachute')
print('mean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma')
for trial in range(4):
	VAE_MODELS_FOLDER = os.path.join(args.robot_ckpt, f'trial{trial}', "models")
	DEFAULT_RESULTS_FOLDER = os.path.dirname(VAE_MODELS_FOLDER)
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'dynamics', "models")
	global_step = 0
	if not os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(VAE_MODELS_FOLDER, 'final.pth')):
		print('Please use the same directory as the final VAE model')
		exit(-1)

	if os.path.exists(os.path.join(MODELS_FOLDER,'hri_hyperparams.npz')):
		hyperparams = np.load(os.path.join(MODELS_FOLDER,'hri_hyperparams.npz'), allow_pickle=True)
		hri_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		hri_config = hyperparams['hri_config'].item()
		config = hyperparams['global_config'].item()
	else:
		print(os.path.join(MODELS_FOLDER,'hyperparams.npz'),' does not exist')
		hri_config = hri_config()
		config = global_config()

	robot_vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	robot_vae_args = robot_vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	robot_vae_config = robot_vae_hyperparams['vae_config'].item()
	robot_vae = networks.VAE(**(robot_vae_config.__dict__)).to(device)
	ckpt = torch.load(os.path.join(VAE_MODELS_FOLDER,'final.pth'))
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
	ckpt = torch.load(os.path.join(MODELS_FOLDER, 'tdm_final.pth'))
	hri.load_state_dict(ckpt['model'])
	hri.eval()

	# with np.load(args.src, allow_pickle=True) as data:
	# 	test_data = [torch.Tensor(traj) for traj in data['test_data']]
	# 	test_iterator = DataLoader(test_data, batch_size=1, shuffle=True)
	from mild_hri.dataloaders import *
	if robot_vae_args.model =='VAE_PEPPER':
		dataset = nuisi.PepperWindowDataset
	elif robot_vae_args.model =='VAE_YUMI':
		dataset = buetepage_hr.YumiWindowDataset

	test_dataset = dataset(robot_vae_args.src, train=False, window_length=config.WINDOW_LEN, downsample=0.2)

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

	pred_mse_ckpt = []
	pred_mse_nowave_ckpt = []
	for mse in pred_mse_actions_ckpt:
		pred_mse_ckpt+= mse
	for mse in pred_mse_actions_ckpt[1:]:
		pred_mse_nowave_ckpt+= mse
	# s = f'{trial}'#{np.mean(pred_mse_ckpt):.4e}\t{np.std(pred_mse_ckpt):.4e}\t{np.mean(pred_mse_nowave_ckpt):.4e}\t{np.std(pred_mse_nowave_ckpt):.4e}'

	pred_mse += pred_mse_ckpt
	pred_mse_nowave += pred_mse_nowave_ckpt
	for i in range(4):
		if best_action_mse[i] > np.mean(pred_mse_actions_ckpt[i]):
			best_action_mse[i] = np.mean(pred_mse_actions_ckpt[i])
			pred_mse_actions[i] = pred_mse_actions_ckpt[i]
s = ''
for mse in pred_mse_actions:
	s += f'\t{np.mean(mse):.3f} $\pm$ {np.std(mse):.3f}'
print(s)
np.savez_compressed('logs/mse/yumi_20hz_3joints_xvel.npz', np.array(pred_mse_actions,dtype=object))
# s = f'all\t{np.mean(pred_mse):.4e}\t{np.std(pred_mse):.4e}\t{np.mean(pred_mse_nowave):.4e}\t{np.std(pred_mse_nowave):.4e}'
# for mse in pred_mse_actions:
# 	s += f'\t{np.mean(mse):.4e}\t{np.std(mse):.4e}'
# print(s)