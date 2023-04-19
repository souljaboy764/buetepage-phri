import torch
import numpy as np
import argparse, os

import networks
from utils import *

parser = argparse.ArgumentParser(description='Buetepage Human VAE Testing')
parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
					help='Checkpoint to test')
parser.add_argument('--src', type=str, default='./data/orig/vae_data.npz', metavar='RES',
					help='Path to read training and testin data (default: ./data/orig/vae_data.npz).')
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Creating and Loading Model")
dirname = os.path.dirname(args.ckpt)
hyperparams = np.load(os.path.join(dirname,'hyperparams.npz'), allow_pickle=True)
train_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
vae_config = hyperparams['vae_config'].item()
ckpt = torch.load(args.ckpt)

model = getattr(networks, train_args.model)(**(vae_config.__dict__)).to(device)
model.load_state_dict(ckpt['model'])

model.eval()

print("Reading Data")
with np.load(args.src, allow_pickle=True) as data:
	test_data = torch.Tensor(np.array(data['test_data']).astype(np.float32)).to(device)
x_gen, zpost_samples, zpost_dist = model(test_data[..., :model.input_dim])

x_gen = x_gen.reshape(test_data.shape[0], model.window_size, model.num_joints, model.joint_dims)
test_data_p1 = test_data[..., :model.input_dim].reshape(test_data.shape[0], model.window_size, model.num_joints, model.joint_dims)
test_data_p2 = test_data[..., model.input_dim:].reshape(test_data.shape[0], model.window_size, model.num_joints, model.joint_dims)
test_data = torch.concat([test_data_p1,test_data_p2], dim=-1)

# error = (test_data - x_gen)**2
# print("Prediction error",error.sum(-1).sum(-1).mean())
x_gen = x_gen.cpu().detach().numpy()
test_data = test_data.cpu().detach().numpy()
# np.savez_compressed('vae_test.npz', x_gen=x_gen, test_data=test_data)

fig, ax = prepare_axis()
for frame_idx in range(test_data.shape[0]):
	ax = reset_axis(ax)
	ax = visualize_skeleton(ax, test_data[frame_idx, ..., :3], color='r', linestyle='-')
	ax = visualize_skeleton(ax, x_gen[frame_idx], color='b', linestyle='--')
	plt.pause(0.01)
	if not plt.fignum_exists(fig.number):
		break
	
plt.ioff()
plt.show()