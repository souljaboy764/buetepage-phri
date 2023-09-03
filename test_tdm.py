import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *

import numpy as np
import os, argparse

import networks
from utils import *

parser = argparse.ArgumentParser(description='SKID Training')
parser.add_argument('--vae-ckpt', type=str, metavar='CKPT', default='logs/vae_hh_orig_oldcommit_AdamW_07011535_tdmfixed/models/final.pth',
					help='Path to the VAE checkpoint, where the TDM models will also be saved.')
parser.add_argument('--src', type=str, default='./data/orig_bothactors/tdm_data.npz', metavar='DATA',
					help='Path to read training and testin data (default: ./data/orig_bothactors/tdm_data.npz).')
args = parser.parse_args()
torch.manual_seed(128542)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(args.vae_ckpt)))
VAE_MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, 'tdm', "models")
if not os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(VAE_MODELS_FOLDER, 'final.pth')):
	print('Please use the same directory as the final VAE model')
	exit(-1)

if os.path.exists(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz')):
	hyperparams = np.load(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz'), allow_pickle=True)
	args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	tdm_config = hyperparams['tdm_config'].item()
else:
	print('No TDM configs found')
	exit(-1)

vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
vae_args = vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
vae_config = vae_hyperparams['vae_config'].item()

print("Creating Model and Optimizer")
tdm_1 = networks.TDM(**(tdm_config.__dict__)).to(device)
tdm_2 = networks.TDM(**(tdm_config.__dict__)).to(device)

if os.path.exists(os.path.join(MODELS_FOLDER, 'tdm_final.pth')):
	print("Loading Checkpoints")
	ckpt = torch.load(os.path.join(MODELS_FOLDER, 'tdm_final.pth'))
	tdm_1.load_state_dict(ckpt['model_1'])
	tdm_2.load_state_dict(ckpt['model_2'])
	global_step = ckpt['epoch']
else:
	print('No TDM model found')
	exit(-1)

vae = getattr(networks, vae_args.model)(**(vae_config.__dict__)).to(device)
ckpt = torch.load(args.vae_ckpt)
vae.load_state_dict(ckpt['model'])
vae.eval()

print("Reading Data")
with np.load(args.src, allow_pickle=True) as data:
	test_data_np = data['test_data']
	test_data = [torch.Tensor(traj) for traj in test_data_np]
	test_num = len(test_data)
	print(test_num,'Testing Trajecotries')
	lens = []
	for traj in test_data:
		lens.append(traj.shape[0])

	# padded_sequences = pad_sequence(test_data, batch_first=True, padding_value=1.)
	# test_iterator = DataLoader(test_data, batch_size=1, shuffle=False)
# p1_tdm_idx = np.concatenate([np.arange(12),np.arange(-5,0)])
# p2_tdm_idx = np.concatenate([480+np.arange(12),np.arange(-5,0)])
# p1_vae_idx = np.arange(480)
# p2_vae_idx = np.arange(480) + 480

p1_tdm_idx = np.concatenate([np.arange(18),np.arange(-5,0)])
p2_tdm_idx = np.concatenate([90+np.arange(18),np.arange(-5,0)])
p1_vae_idx = np.arange(90)
p2_vae_idx = np.arange(90) + 90
actidx = np.array([[0,7],[7,15],[15,29],[29,39]])

print("Starting Evaluation")
total_loss_1 = []
total_loss_2 = []
x1_vae_gen = []
x2_vae_gen = []
x_tdm_gen = []
mse_actions = []

print('Pred. MSE (all)\t\tPred. MSE w/o waving\t\tPred. MSE waving\t\tPred. MSE handshake\t\tPred. MSE rocket\t\tPred. MSE parachute')
print('mean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma')

for a in actidx:
	mse_actions.append([])
	for i in range(a[0],a[1]):
		x = test_data[i]
		seq_len, dims = x.shape
		x1_tdm = x[None,:,p1_tdm_idx].to(device)
		x2_vae = x[None,:,p2_vae_idx].to(device)
		
		# z1_d1_dist, d1_samples, d1_dist = tdm_1(x1_tdm, lens)
		d1_x1 = tdm_1.latent_mean(tdm_1.activation(tdm_1._encoder(x1_tdm)[0]))
		z2_d1 = tdm_2.output_mean(tdm_2._decoder(d1_x1))
		x2_tdm_out = vae._output(vae._decoder(z2_d1))

		mse_actions[-1] += ((x2_tdm_out - x2_vae)**2).reshape((seq_len, vae.window_size, vae.num_joints, vae.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()

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