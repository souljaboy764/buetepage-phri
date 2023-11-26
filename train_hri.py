import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, argparse

import networks
from config import *
from utils import *

from phd_utils.dataloaders import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_iters_hri(iterator:DataLoader, hri:networks.HRIDynamics, robot_vae:networks.VAE, human_tdm:networks.TDM, optimizer:torch.optim.Optimizer):
	iters = 0
	total_loss = []
	total_recon_loss = []
	total_kl_loss = []
	for i, (x, label) in enumerate(iterator):
		if hri.training:
			optimizer.zero_grad()
		x = torch.Tensor(x[0])
		label = torch.Tensor(label[0])
		x = torch.cat([x,label], dim=-1)
		seq_len, dims = x.shape
		x_p1_tdm = x[:,p1_tdm_idx].to(device)
		x_r2_vae = x[:,r2_vae_idx].to(device)
		with torch.no_grad():
			_, _, d_x1_dist, _ = human_tdm(x_p1_tdm, None)
			z_r2vae_dist = robot_vae(x_r2_vae, True)

		x_r2_hri = torch.concat([x[:,r2_hri_idx].to(device), d_x1_dist.mean], dim=-1)
		z_r2hri_dist, z_r2hri_samples, _ = hri(x_r2_hri, None)

		x_r2vaehri_gen = robot_vae._output(robot_vae._decoder(z_r2hri_samples))
		
		recon_loss =  F.mse_loss(x_r2vaehri_gen,x_r2_vae,reduction='none').sum()
		kl_loss = torch.distributions.kl_divergence(z_r2hri_dist, z_r2vae_dist).sum()
		loss = recon_loss + robot_vae.beta*kl_loss
		total_loss.append(loss)
		total_recon_loss.append(recon_loss)
		total_kl_loss.append(kl_loss)

		if hri.training:
			loss.backward()
			# nn.utils.clip_grad_norm_(hri.parameters(), 1.0)
			optimizer.step()
		iters += 1

	return total_loss,total_recon_loss,total_kl_loss, x_r2vaehri_gen.reshape(-1, robot_vae.window_size, robot_vae.num_joints, robot_vae.joint_dims), z_r2hri_samples, z_r2vae_dist.sample(), x_r2_vae.reshape(-1, robot_vae.window_size, robot_vae.num_joints, robot_vae.joint_dims), iters

def write_summaries_hri(writer, loss, recon_loss, kl_loss, x_gen, x, z_r2hri_samples, z_r2vae_samples, steps_done, prefix):
	writer.add_scalar(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon_loss), steps_done)
	writer.add_scalar(prefix+'/kl_loss', sum(kl_loss), steps_done)

	writer.add_histogram(prefix+'latents/z_r2vae_samples', z_r2hri_samples.mean(0), steps_done)
	writer.add_histogram(prefix+'latents/z_r2vae_samples', z_r2vae_samples.mean(0), steps_done)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Training Human-Robot Interactive Dynamics')
	parser.add_argument('--human-ckpt', type=str, metavar='HUMAN-CKPT', required=True,
						help='Path to the Human TDM checkpoint.')
	parser.add_argument('--robot-ckpt', type=str, metavar='ROBOT-CKPT', required=True,
						help='Path to the robot VAE checkpoint, where the TDM models will also be saved.')
	args = parser.parse_args()
	seed = np.random.randint(0,np.iinfo(np.int32).max)
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

	robot_vae_hyperparams = np.load(os.path.join(VAE_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	robot_vae_args = robot_vae_hyperparams['args'].item() # overwrite args if loading from checkpoint
	robot_vae_config = robot_vae_hyperparams['config'].item()
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

	config = global_config()
	hri_config = hri_config()

	print("Reading Data")
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
	
	train_dataset = dataset(train=True, window_length=config.WINDOW_LEN, downsample=0.2)
	test_dataset = dataset(train=False, window_length=config.WINDOW_LEN, downsample=0.2)
	
	train_dataset.labels = []
	for idx in range(len(train_dataset.actidx)):
		for i in range(train_dataset.actidx[idx][0], train_dataset.actidx[idx][1]):
			label = np.zeros((train_dataset.traj_data[i].shape[0], len(train_dataset.actidx)))
			label[:, idx] = 1
			train_dataset.labels.append(label)

	test_dataset.labels = []
	for idx in range(len(test_dataset.actidx)):
		for i in range(test_dataset.actidx[idx][0], test_dataset.actidx[idx][1]):
			label = np.zeros((test_dataset.traj_data[i].shape[0], len(test_dataset.actidx)))
			label[:, idx] = 1
			test_dataset.labels.append(label)

	train_iterator = DataLoader(train_dataset, batch_size=1, shuffle=True)
	test_iterator = DataLoader(test_dataset, batch_size=1, shuffle=True)

	print("Creating Model and Optimizer")
	hri = networks.HRIDynamics(**(hri_config.__dict__)).to(device)
	optimizer = getattr(torch.optim, config.optimizer)(hri.parameters(), lr=config.lr)
	np.savez_compressed(os.path.join(MODELS_FOLDER,'hri_hyperparams.npz'), args=args, global_config=config, hri_config=hri_config)

	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	s = ''
	for k in config.__dict__:
		s += str(k) + ' : ' + str(config.__dict__[k]) + '\n'
	writer.add_text('global_config', s)

	s = ''
	for k in hri_config.__dict__:
		s += str(k) + ' : ' + str(hri_config.__dict__[k]) + '\n'
	writer.add_text('hri_config', s)

	writer.flush()

	print("Starting Epochs")
	for epoch in range(config.EPOCHS):
		hri.train()
		train_loss, train_recon_loss, train_kl_loss, x_r2vaehri_gen, z_r2hri_samples, z_r2vae_samples, x, iters = run_iters_hri(train_iterator, hri, robot_vae, human_tdm, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_hri(writer, train_loss, train_recon_loss, train_kl_loss, x_r2vaehri_gen, x, z_r2hri_samples, z_r2vae_samples, steps_done, 'train')
		for name, param in hri.named_parameters():
			if param.grad is None:
				continue
			value = param.reshape(-1)
			grad = param.grad.reshape(-1)
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		if epoch % config.EPOCHS_TO_SAVE == 0:
			hri.eval()
			with torch.no_grad():
				test_loss, test_recon_loss, test_kl_loss, x_r2vaehri_gen, z_r2hri_samples, z_r2vae_samples, x, iters = run_iters_hri(test_iterator, hri, robot_vae, human_tdm, optimizer)
				write_summaries_hri(writer, test_loss, test_recon_loss, test_kl_loss, x_r2vaehri_gen, x, z_r2hri_samples, z_r2vae_samples, steps_done, 'test')

			checkpoint_file = os.path.join(MODELS_FOLDER, 'hri_%0.4d.pth'%(epoch))
			torch.save({'model': hri.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'hri_final.pth')
	torch.save({'model': hri.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step}, checkpoint_file)
