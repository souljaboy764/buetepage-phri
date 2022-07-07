import torch

import numpy as np
import argparse, os

import networks

parser = argparse.ArgumentParser(description='Buetepage Human VAE Testing')
parser.add_argument('--ckpt', type=str, default='logs/vae_hh_orig_oldcommit_AdamW_07011535/models/final.pth', metavar='CKPT',
					help='Checkpoint to test')
parser.add_argument('--src', type=str, default='./data/orig/vae/data.npz', metavar='RES',
					help='Path to read training and testin data (default: ./data/orig/vae/data.npz).')
args = parser.parse_args()
torch.manual_seed(128542)
np.random.seed(19680801)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Reading Data")
with np.load(args.src, allow_pickle=True) as data:
	test_data = torch.Tensor(np.array(data['test_data']).astype(np.float32)).to(device)

print("Creating and Loading Model")
dirname = os.path.dirname(args.ckpt)
hyperparams = np.load(os.path.join(dirname,'hyperparams.npz'), allow_pickle=True)
train_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
vae_config = hyperparams['vae_config'].item()
ckpt = torch.load(args.ckpt)

model = getattr(networks, train_args.model)(**(vae_config.__dict__)).to(device)
model.load_state_dict(ckpt['model'])

model.eval()
x_gen, zpost_samples, zpost_dist = model(test_data)
x_gen = x_gen.reshape(-1, model.window_size, model.num_joints, model.joint_dims)
test_data = test_data.reshape(-1, model.window_size, model.num_joints, model.joint_dims)
error = (test_data - x_gen)**2
print("Prediction error",error.sum(-1).sum(-1).mean())
x_gen = x_gen.cpu().detach().numpy()
test_data = test_data.cpu().detach().numpy()
np.savez_compressed('vae_test.npz', x_gen=x_gen, test_data=test_data)
