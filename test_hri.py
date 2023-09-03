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
os.makedirs(MODELS_FOLDER,exist_ok=True)
SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'dynamics', "summary")
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

# print("Reading Data")
# with np.load(args.src, allow_pickle=True) as data:
	
# 	test_data_np = data['test_data']
# 	test_data = [torch.Tensor(traj) for traj in test_data_np]
# 	test_num = len(test_data)
# 	lens = []
# 	for traj in test_data:
# 		lens.append(traj.shape[0])

# 	padded_sequences = pad_sequence(test_data, batch_first=True, padding_value=1.)
# 	test_iterator = DataLoader(list(zip(padded_sequences, lens)), batch_size=len(test_data), shuffle=False)
# print("Reading Data")
# test_data = [torch.Tensor(traj) for traj in dataloaders.buetepage_hr.SequenceWindowDataset(args.src, train=False, window_length=config.WINDOW_LEN).traj_data]
# test_num = len(test_data)
# lens = []
# for traj in test_data:
# 	lens.append(traj.shape[0])

# padded_sequences = pad_sequence(test_data, batch_first=True, padding_value=1.)
# test_iterator = DataLoader(list(zip(padded_sequences,lens)), batch_size=len(padded_sequences), shuffle=True)

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

print("Starting Evaluation")
total_loss = []
x_gen = []
for i, x in enumerate(test_iterator):
	x = x[0]
	seq_len, dims = x.shape
	x_p1_tdm = x[:,p1_tdm_idx].to(device)
	x_r2_gt = x[:,r2_vae_idx].to(device)
	x_r2_hri = x[:,r2_hri_idx].to(device)
	with torch.no_grad():
		_, _, d_x1_dist = human_tdm(x_p1_tdm, None)
		hri_input = torch.concat([x_r2_hri, d_x1_dist.mean], dim=-1)
		z_r2hri_dist, z_r2hri_samples = hri(hri_input, None)
		x_r2_gen = robot_vae._output(robot_vae._decoder(z_r2hri_dist.mean))
				
	loss = F.mse_loss(x_r2_gt, x_r2_gen, reduction='none')
	total_loss.append(loss.detach().cpu().numpy())
	x_gen.append(x_r2_gen.detach().cpu().numpy())

total_loss = np.concatenate(total_loss,axis=0)
print(total_loss.shape)
np.savez_compressed(os.path.join(DEFAULT_RESULTS_FOLDER,'recon_error.npz'), error=total_loss, lens=lens)
# x_gen = np.concatenate(x_gen,axis=0)
x_gen = np.array(x_gen)
print(total_loss.mean())
print(np.shape(x_gen))
print(np.shape(padded_sequences.cpu().detach().numpy()))
print(os.path.join(DEFAULT_RESULTS_FOLDER,'hri_test.npz'))
np.savez_compressed(os.path.join(DEFAULT_RESULTS_FOLDER,'hri_test.npz'), x_gen=x_gen, test_data=padded_sequences.cpu().detach().numpy(), lens=lens)