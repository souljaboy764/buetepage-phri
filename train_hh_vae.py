import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

from networks import VAE
from config import global_config, human_vae_config
from utils import *

parser = argparse.ArgumentParser(description='SKID Training')
parser.add_argument('--results', type=str, default='./logs/results/'+datetime.datetime.now().strftime("%m%d%H%M"), metavar='RES',
					help='Path for saving results (default: ./logs/results/MMDDHHmm).')
parser.add_argument('--src', type=str, default='./data/orig/vae/data.npz', metavar='RES',
					help='Path to read training and testin data (default: ./data/orig/vae/data.npz).')
args = parser.parse_args()
torch.manual_seed(128542)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Creating Model and Optimizer")
config = global_config()
vae_config = human_vae_config()
model = VAE(**(vae_config.__dict__)).to(device)
optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)

print("Reading Data")
with np.load(args.src, allow_pickle=True) as data:
	train_data = np.array(data['train_data']).astype(np.float32)
	test_data = np.array(data['test_data']).astype(np.float32)
	train_iterator = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
	test_iterator = DataLoader(test_data, batch_size=model.batch_size, shuffle=True)

DEFAULT_RESULTS_FOLDER = args.results
MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
global_step = 0
if os.path.exists(DEFAULT_RESULTS_FOLDER) and os.path.exists(os.path.join(MODELS_FOLDER, 'final.pth')):
	print("Loading Checkpoints")
	ckpt = torch.load(os.path.join(MODELS_FOLDER, 'final.pth'))
	model.load_state_dict(ckpt['model'])
	optimizer.load_state_dict(ckpt['optimizer'])
	global_step = ckpt['epoch']
elif not os.path.exists(DEFAULT_RESULTS_FOLDER):
	print("Creating Result Directories")
	os.makedirs(DEFAULT_RESULTS_FOLDER)
	os.makedirs(MODELS_FOLDER)
	os.makedirs(SUMMARIES_FOLDER)

print("Building Writer")
writer = SummaryWriter(SUMMARIES_FOLDER)
# x_gen,_,_  = model(torch.Tensor(test_data[:10]).to(device))
model.eval()
writer.add_graph(model, torch.Tensor(test_data[:10]).to(device))
model.train()
s = ''
for k in config.__dict__:
	s += str(k) + ' : ' + str(config.__dict__[k]) + '\n'
writer.add_text('global_config', s)

s = ''
for k in vae_config.__dict__:
	s += str(k) + ' : ' + str(vae_config.__dict__[k]) + '\n'
writer.add_text('human_vae_config', s)

writer.flush()

print("Starting Epochs")
for epoch in range(config.EPOCHS):
	model.train()
	train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iters(train_iterator, model, optimizer)
	steps_done = (epoch+1)*iters
	write_summaries(writer, train_recon, train_kl, train_loss, x_gen, zx_samples, x, steps_done, 'train')
	params = []
	grads = []
	for name, param in model.named_parameters():
		writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
		writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
		if torch.allclose(param.grad, torch.zeros_like(param.grad)):
			print('zero grad for',name)
	
	model.eval()
	with torch.no_grad():
		test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iters(test_iterator, model, optimizer)
		write_summaries(writer, test_recon, test_kl, test_loss, x_gen, zx_samples, x, steps_done, 'test')

	if epoch % config.EPOCHS_TO_SAVE == 0:
		checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
		torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

	print(epoch,'epochs done')

writer.flush()

checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
