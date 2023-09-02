import numpy as np
import os
import argparse

from human_robot_interaction_data.read_hh_hr_data import *

from utils import downsample_trajs

def vae_tdm_preproc(trajectories, labels, window_length=40, robot=False):
	vae_inputs = []
	sequences = []
	for i in range(len(trajectories)):
		trajs_concat = []
		if robot:
			for traj in [trajectories[i][:,:12], trajectories[i][:,12:]]:
				traj_shape = traj.shape
				idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - window_length)])
				trajs_concat.append(traj[idx].reshape((traj_shape[0] + 1 - window_length, window_length*traj_shape[-1])))
		else:
			for traj in [trajectories[i][:,:,:3], trajectories[i][:,:,3:]]:
				traj_shape = traj.shape
				if len(traj_shape)!=3 and traj_shape[1]*traj_shape[2]!=12:
					print('Skipping trajectory not conforming to Dimensions LENx4x3')
					continue
				
				idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - window_length)])
				trajs_concat.append(traj[idx].reshape((traj_shape[0] + 1 - window_length, window_length*4*3)))

		trajs_concat = np.concatenate(trajs_concat,axis=-1)
		if i == 0:
			vae_inputs = trajs_concat
		else:
			vae_inputs = np.vstack([vae_inputs, trajs_concat])
		sequences.append(np.concatenate([trajs_concat,labels[i][:trajs_concat.shape[0]]],axis=-1))

	return np.array(vae_inputs), np.array(sequences)

def preproc(src_dir, robot=False):
	train_data, train_labels, test_data, test_labels = [], [], [] ,[]
	action_onehot = np.eye(5)
	actions = ['hand_wave', 'hand_shake', 'rocket', 'parachute']
	
	for a in range(len(actions)):
		action = actions[a]
		trajectories, traj_labels = [], []
		idx_list = np.array([joints_dic[joint] for joint in ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']])
		trials = ['1'] if robot else ['1', '2']
		for trial in trials:
			if robot:
				data_p1, data_r2 = read_hri_data(action, os.path.join(src_dir, 'hr'))
				segments_file_h = os.path.join(src_dir, 'hr', 'segmentation', action+'_p1.npy')
				segments_file_r = os.path.join(src_dir, 'hr', 'segmentation', action+'_r2.npy')
				segments = np.load(segments_file_h)
				segments_r = np.load(segments_file_r)

			else:
				data_file_p1 = os.path.join(src_dir, 'hh','p1',action+'_s1_'+trial+'.csv')
				data_file_p2 = os.path.join(src_dir, 'hh','p2',action+'_s2_'+trial+'.csv')
				data_p2, _, _, _ = read_data(data_file_p2)
				data_p2[..., [0,1,2]]  = data_p2[..., [2,0,1]]
				data_p2[..., 1] *= -1
				segment_file = os.path.join(src_dir, 'hh', 'segmentation', action+'_'+trial+'.npy')
				segments = np.load(segment_file)
			
				data_p1, _, _, _ = read_data(data_file_p1)
			data_p1[..., [0,1,2]]  = data_p1[..., [2,0,1]]
			data_p1[..., 1] *= -1
	
			for i in range(len(segments)):
				s = segments[i]
				traj1 = data_p1[s[0]:s[1], idx_list] # seq_len, 4 ,3
				labels = np.zeros((traj1.shape[0],5))
				labels[:] = action_onehot[a]
				if robot:
					s_r = segments_r[i]
					traj2 = data_r2[s_r[0]:s_r[1]] # seq_len, 7
					traj1 = traj1 - traj1[0,0]
					
					# the indices where no movement occurs at the end are annotated as "not active". (Sec. 4.3.1 of the paper)
					notactive_idx = np.where(np.sqrt(np.power(np.diff(traj1, axis=0),2).sum((2))).mean(1) > 5e-4)[0]
					if len(notactive_idx) > 0:
						labels[notactive_idx[-1]:] = action_onehot[-1]

					traj = np.concatenate([traj1.reshape(-1, len(idx_list)*3), traj2], axis=-1) # seq_len, 19
					
				else:
					traj2 = data_p2[s[0]:s[1], idx_list] # seq_len, 4 ,3
					traj = np.concatenate([traj1, traj2], axis=-1)
					traj = traj - traj[0,0]
			
					# the indices where no movement occurs at the end are annotated as "not active". (Sec. 4.3.1 of the paper)
					notactive_idx = np.where(np.sqrt(np.power(np.diff(traj, axis=0),2).sum((2))).mean(1) > 1e-3)[0]
					if len(notactive_idx) > 0:
						labels[notactive_idx[-1]:] = action_onehot[-1]
				trajectories.append(traj)
				traj_labels.append(labels)
				
		# the first 80% are for training and the last 20% are for testing (Sec. 4.3.2)
		split_idx = int(0.8*len(trajectories))
		train_data += trajectories[:split_idx]
		test_data += trajectories[split_idx:]
		train_labels += traj_labels[:split_idx]
		test_labels += traj_labels[split_idx:]
	
	train_data = np.array(train_data, dtype=object)
	test_data = np.array(test_data, dtype=object)
	train_labels = np.array(train_labels, dtype=object)
	test_labels = np.array(test_labels, dtype=object)
	print('Sequences: Training', train_data.shape, 'Testing', test_data.shape)
	print('Labels: Training', train_labels.shape, 'Testing', test_labels.shape)
	
	return train_data, train_labels, test_data, test_labels

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Data preprocessing for Right arm trajectories of Buetepage et al. (2020).')
	parser.add_argument('--src-dir', type=str, default='./human_robot_interaction_data', metavar='SRC',
						help='Path where https://github.com/souljaboy764/human_robot_interaction_data is extracted to read csv files (default: ./human_robot_interaction_data).')
	parser.add_argument('--dst-dir', type=str, default='/tmp/data/', metavar='DST',
						help='Path to save the processed trajectories to (default: ./data).')
	parser.add_argument('--downsample-len', type=int, default=0, metavar='NEW_LEN',
						help='Length to downsample trajectories to. If 0, no downsampling is performed (default: 0).')
	parser.add_argument('--robot', action='store_true', default=False,
						help='Whether to use HRI or HHI data (default: False -> HHI)')
	args = parser.parse_args()
	
	train_data, train_labels, test_data, test_labels = preproc(args.src_dir, args.robot)

	if not os.path.exists(args.dst_dir):
		os.mkdir(args.dst_dir)

	if args.downsample_len == 0:
		np.savez_compressed(os.path.join(args.dst_dir, 'traj_data.npz'), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
	else:
		train_data, train_labels = downsample_trajs(train_data, train_labels, args.downsample_len)
		np.savez_compressed(os.path.join(args.dst_dir, 'labelled_sequences_augmented.npz'), data=train_data, labels=train_labels)
		args.robot = False
	
	# vae_train_data, tdm_train_data = vae_tdm_preproc(train_data, train_labels, robot=args.robot)
	# vae_test_data, tdm_test_data = vae_tdm_preproc(test_data, test_labels, robot=args.robot)
	# print('VAE Data: Training',vae_train_data.shape, 'Testing', vae_test_data.shape)
	# print('TDM Data: Training',tdm_train_data.shape, 'Testing', tdm_test_data.shape)
	# np.savez_compressed(os.path.join(args.dst_dir,'vae_data.npz'), train_data=vae_train_data, test_data=vae_test_data)
	# np.savez_compressed(os.path.join(args.dst_dir,'tdm_data.npz'), train_data=tdm_train_data, test_data=tdm_test_data)
		
