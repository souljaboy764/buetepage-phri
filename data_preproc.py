import torch
from torch.nn. functional import grid_sample, affine_grid

import numpy as np
import os
import argparse

from human_robot_interaction_data.read_hh_hr_data import read_data, joints_dic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vae_tdm_preproc(trajectories, labels, window_length=40):	
	vae_inputs = []
	sequences = []
	seq_lens = []
	for i in range(len(trajectories)):
		trajs_concat = []
		seq_len, num_joints, dim = trajectories[i].shape
		for traj in [trajectories[i][:,:,:3], trajectories[i][:,:,3:]]:
			traj_shape = traj.shape
			
			# for i in range(traj_shape[0]- 1 + window_length):
			# 	sequences.append(traj[i:i+window_length].flatten())

			idx = np.array([np.arange(i,i+window_length) for i in range(seq_len + 1 - window_length)])
			trajs_concat.append(traj[idx].reshape((seq_len + 1 - window_length, window_length*num_joints*dim//2)))
			print(trajs_concat[-1].shape)
		trajs_concat = np.concatenate(trajs_concat,axis=-1)
		print(trajs_concat.shape)
		print('\n')
		# if i == 0:
		# 	vae_inputs = trajs_concat
		# else:
		# 	vae_inputs = np.vstack([vae_inputs, trajs_concat])
		vae_inputs.append(trajs_concat)
		seq_lens.append(seq_len)
		sequences.append(np.concatenate([trajs_concat,labels[i][:trajs_concat.shape[0]]],axis=-1))
		
	return np.array(vae_inputs), np.array(seq_lens), np.array(sequences)

def preproc(src_dir, downsample_len=250, augment=False):
	
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	
	idx_list = np.array([joints_dic[joint] for joint in ['Hips', 'Spine1', 'Neck', 'Head', 
														'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
														'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']])
	# idx_list = np.array([joints_dic[joint] for joint in ['RightHand']])
	theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(len(idx_list),1,1)
	action_onehot = np.eye(5)
	actions = ['hand_wave', 'hand_shake', 'rocket', 'parachute']
	
	for a in range(len(actions)):
		action = actions[a]
		trajectories = []
		traj_labels = []

		for trial in ['1','2']:
			data_file_p1 = os.path.join(src_dir, 'hh','p1',action+'_s1_'+trial+'.csv')
			data_p1, data_q, names, times = read_data(data_file_p1)

			data_file_p2 = os.path.join(src_dir, 'hh','p2',action+'_s2_'+trial+'.csv')
			data_p2, data_q, names, times = read_data(data_file_p2)
		
			segment_file = os.path.join(src_dir, 'hh', 'segmentation', action+'_'+trial+'.npy')
			segments = np.load(segment_file)

			for s in segments:
				traj1 = data_p1[s[0]:s[1], idx_list] # seq_len, N, 3
				traj2 = data_p2[s[0]:s[1], idx_list] # seq_len, N, 3
				traj = np.concatenate([traj1, traj2], axis=-1)
				traj = traj - traj[0,0]
				# downsample = int((s[1] - s[0])*0.4)
				if downsample_len > 0:
				# if True:
					traj = traj.transpose(1,2,0) # N, 3, seq_len
					traj = torch.Tensor(traj).to(device).unsqueeze(2) # N, 3, 1 seq_len
					traj = torch.concat([traj, torch.zeros_like(traj)], dim=2) # N, 3, 2 seq_len
					
					grid = affine_grid(theta, torch.Size([len(idx_list), 3, 2, downsample_len]), align_corners=True)
					traj = grid_sample(traj.type(torch.float32), grid, align_corners=True) # 4, 3, 2 new_length
					traj = traj[:, :, 0].cpu().detach().numpy() # 4, 3, new_length
					traj = traj.transpose(2,0,1) # new_length, 4, 3
					trajectories.append(traj)
				else:
					trajectories.append(traj)
				
				labels = np.zeros((traj.shape[0],5))
				labels[:] = action_onehot[a]
				
				# the indices where no movement occurs at the end are annotated as "not active". (Sec. 4.3.1 of the paper)
				# notactive_idx = np.where(np.sqrt(np.power(np.diff(traj, axis=0),2).sum((2))).mean(1) > 1e-3)[0]
				# labels[notactive_idx[-1]:] = action_onehot[-1]
				
				traj_labels.append(labels)
		# train_data += trajectories[:26] # in order to balance the number of samples of each action, data would any be augmented next.
		# test_data += trajectories[26:31]
		
		# the first 80% are for training and the last 20% are for testing (Sec. 4.3.2)
		split_idx = int(0.8*len(trajectories))
		train_data += trajectories[:split_idx]
		test_data += trajectories[split_idx:]
		train_labels += traj_labels[:split_idx]
		test_labels += traj_labels[split_idx:]
	
	train_data = np.array(train_data)
	test_data = np.array(test_data)
	train_labels = np.array(train_labels)
	test_labels = np.array(test_labels)
	print('Sequences: Training',train_data.shape, 'Testing', test_data.shape)
	print('Labels: Training',train_labels.shape, 'Testing', test_labels.shape)
	
	if augment: # Augment only if downsampling the trajectories
		M = np.eye(downsample_len)*2
		for i in range(1,downsample_len):
			M[i][i-1] = M[i-1][i] = -1

		B = torch.Tensor(np.linalg.pinv(M) * 5e-5).to(device)
		L = torch.linalg.cholesky(B) # faster/momry-friendly to directly give cholesky
		num_trajs = len(train_data)
		n_augments = 70
		for i in range(num_trajs):
			augments = torch.distributions.MultivariateNormal(torch.Tensor(train_data[i]).to(device).reshape(len(idx_list)*6, downsample_len), scale_tril=L).sample((n_augments,)).cpu().numpy()
			train_data = np.concatenate([train_data,augments.reshape(n_augments, downsample_len, len(idx_list), 6)], 0)
			train_labels = np.concatenate([train_labels,np.repeat(train_labels[i:i+1], augments.shape[0], axis=0)], 0)
		print('Augmented Sequences: Training',train_data.shape, 'Testing', test_data.shape)
		print('Augmented Labels: Training',train_labels.shape, 'Testing', test_labels.shape)
	return train_data, train_labels, test_data, test_labels

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Data preprocessing for Right arm trajectories of Buetepage et al. (2020).')
	parser.add_argument('--src-dir', type=str, default='./human_robot_interaction_data', metavar='SRC',
						help='Path where https://github.com/souljaboy764/human_robot_interaction_data is extracted to read csv files (default: ./human_robot_interaction_data).')
	parser.add_argument('--dst-dir', type=str, default='./data/ae_bip/', metavar='DST',
						help='Path to save the processed trajectories to (default: ./data).')
	parser.add_argument('--downsample-len', type=int, default=0, metavar='NEW_LEN',
						help='Length to downsample trajectories to. If 0, no downsampling is performed (default: 0).')
	parser.add_argument('--augment', action="store_true",
						help='Whether to skip the trajectory augmentation or not. (default: False).')
	parser.add_argument('--window-size', type=int, default=40, metavar='WINDOW',
						help='Temporal Window size (default: 40).')
	args = parser.parse_args()
	
	train_data, train_labels, test_data, test_labels = preproc(args.src_dir, args.downsample_len, args.augment)

	if args.dst_dir is not None:
		if not os.path.exists(args.dst_dir):
			os.mkdir(args.dst_dir)

		# if args.downsample_len == 0:
			np.savez_compressed(os.path.join(args.dst_dir, 'labelled_sequences.npz'), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
			vae_train_data, seq_lens_train, tdm_train_data = vae_tdm_preproc(train_data, train_labels, args.window_len)
			vae_test_data, seq_lens_test, tdm_test_data = vae_tdm_preproc(test_data, test_labels, args.window_len)
			print('VAE Data: Training',vae_train_data.shape, 'Testing', vae_test_data.shape)
			print('TDM Data: Training',tdm_train_data.shape, 'Testing', tdm_test_data.shape)
			np.savez_compressed(os.path.join(args.dst_dir,'vae_data.npz'), train_data=vae_train_data, test_data=vae_test_data)
			np.savez_compressed(os.path.join(args.dst_dir,'tdm_data.npz'), train_data=tdm_train_data, test_data=tdm_test_data)
		else:
			print('Data Data: Training',train_data.shape, 'Testing', test_data.shape)
			if args.augment:
					filename = 'labelled_sequences_augmented.npz'
			else:
					filename = 'labelled_sequences_'+str(args.downsample_len)+'.npz'
			np.savez_compressed(os.path.join(args.dst_dir, filename), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
			vae_train_data, seq_lens_train, tdm_train_data = vae_tdm_preproc(train_data, train_labels, args.window_len)
			vae_test_data, seq_lens_test, tdm_test_data = vae_tdm_preproc(test_data, test_labels, args.window_len)
			print('VAE Data: Training',vae_train_data.shape, 'Testing', vae_test_data.shape)
			print('TDM Data: Training',tdm_train_data.shape, 'Testing', tdm_test_data.shape)
			np.savez_compressed(os.path.join(args.dst_dir,'vae_data.npz'), train_data=(vae_train_data, seq_lens_train), test_data=(vae_test_data, seq_lens_test))
			np.savez_compressed(os.path.join(args.dst_dir,'tdm_data.npz'), train_data=tdm_train_data, test_data=tdm_test_data)
