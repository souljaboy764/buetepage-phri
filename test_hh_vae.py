import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.animation import FFMpegWriter
import argparse, os
from joblib import Parallel, delayed
import networks
from config import human_vae_config

parser = argparse.ArgumentParser(description='SKID Training')
parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
					help='Checkpoint to test')
parser.add_argument('--src', type=str, default='./data/single_sample_per_action/vae/data.npz', metavar='RES',
					help='Path to read training and testin data (default: ./data/data/single_sample_per_action/data.npz).')
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
x_gen = x_gen.reshape(-1, model.window_size, model.num_joints, model.joint_dims).cpu().detach().numpy()
test_data = test_data.reshape(-1, model.window_size, model.num_joints, model.joint_dims).cpu().detach().numpy()

metadata = dict(title='Buetepage VAE Test', artist='Matplotlib',
                comment='3D visualization of right arm')
writer = FFMpegWriter(fps=40, metadata=metadata)

fig = plt.figure()
ax = fig.add_subplot(projection="3d",title='reconstruction')
ax.set(xlim3d=(-0.5, 0.5), xlabel='X')
ax.set(ylim3d=(-0.5, 0.5), ylabel='Y')
ax.set(zlim3d=(-0.5, 0.5), zlabel='Z')

current_point_recon, = ax.plot([], [], [], '-ko',  mfc='blue', mec='k')
window_points_recon, = ax.plot([], [], [], '--ko', mfc='blue', mec='k', alpha=0.3)

current_point_gt, = ax.plot([], [], [], '-ko', mfc='green', mec='k')

def set_points_and_edges(edges_plot, points): # points.shape == N, 3
    edges_plot.set_data(points[:,:2].T)
    edges_plot.set_3d_properties(points[:,2])

with writer.saving(fig, "writer_test_window.mp4", 300):
    for i in range(x_gen.shape[0]):
        set_points_and_edges(current_point_recon, x_gen[i,0])
        set_points_and_edges(current_point_gt, test_data[i,0])
        Parallel(n_jobs=model.window_size//8)(delayed(set_points_and_edges)(window_points_recon, x_gen[i,j]) for j in range(1, model.window_size, model.window_size//8))
        writer.grab_frame()
        print(i)