import numpy as np

import torch
from torch.nn import functional as F
import os
import argparse
from human_robot_interaction_data.read_hh_hr_data import read_data, joints_dic


def preproc(src_dir, dst_dir=None, crop_zerovel=False, no_augment=False, downsample_len=250):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	data_actions = {}
	new_length = 250
	theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(4,1,1)
	train_data = []
	test_data = []
	for action in ['hand_wave', 'hand_shake', 'rocket', 'parachute']:
		data_actions[action] = []
		for trial in ['1','2']:
			data_file = os.path.join(src_dir, 'hh','p1',action+'_s1_'+trial+'.csv')
			segment_file = os.path.join(src_dir, 'hh', 'segmentation', action+'_'+trial+'.npy')
			data_p, data_q, names, times = read_data(data_file)
		
			segments = np.load(segment_file)
			for s in segments:
				idx_list = np.array([joints_dic[joint] for joint in ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']])
				in_traj = data_p[s[0]:s[1], idx_list] # seq_len, 4 ,3

				if crop_zerovel:
					idx = np.where(np.sqrt(np.power(np.diff(in_traj, axis=0),2).sum((1,2))) > 1e-3)[0]
					in_traj = in_traj[idx[0]:idx[-1]]

				if downsample_len > 0:
					in_traj = in_traj.transpose(1,2,0) # 4, 3, seq_len
					traj = torch.Tensor(in_traj).to(device).unsqueeze(2) # 4, 3, 1 seq_len
					traj = torch.concat([traj, torch.zeros_like(traj)], dim=2) # 4, 3, 2 seq_len
					
					grid = F.affine_grid(theta, torch.Size([4, 3, 2, downsample_len]), align_corners=True)
					window = F.grid_sample(traj.type(torch.float32), grid, align_corners=True) # 4, 3, 2 new_length
					window = window[:, :, 0].cpu().detach().numpy() # 4, 3, new_length
					data_actions[action].append(window.transpose(2,0,1))
				else:
					data_actions[action].append(window.transpose(2,0,1))
		train_data += data_actions[action][:26] # in order to balance the number of samples of each action, data would any be augmented next.
		test_data += data_actions[action][26:31]

	train_data = np.array(train_data)
	test_data = np.array(test_data)
	
	if not no_augment:
		M = np.eye(train_data.shape[1])*2
		for i in range(1,train_data.shape[1]):
			M[i][i-1] = M[i-1][i] = -1

		B = torch.Tensor(np.linalg.pinv(M) * 1e-6).to(device)

		L = torch.linalg.cholesky(B)
		for traj in train_data:
			augments = torch.distributions.MultivariateNormal(torch.Tensor(traj).to(device).reshape(12,250), scale_tril=L).sample((100,)).cpu().numpy()
			train_data = np.concatenate([train_data,augments.reshape(100, 250, 4, 3)], 0)

	if dst_dir is not None:
		if not os.path.exists(dst_dir):
			os.mkdir(dst_dir)

		if no_augment:
			np.savez_compressed(os.path.join(dst_dir, 'buetepage_rarm_trainData.npz'), train_data)
		else:
			np.savez_compressed(os.path.join(dst_dir, 'buetepage_rarm_trainData-10k.npz'), train_data)

		np.savez_compressed(os.path.join(dst_dir, 'buetepage_rarm_testData.npz'), test_data)
	
	return train_data, test_data

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Data preprocessing for Right arm trajectories of Buetepage et al. (2020).')
	parser.add_argument('--src-dir', type=str, default='./human_robot_interaction_data', metavar='SRC',
						help='Path where https://github.com/souljaboy764/human_robot_interaction_data is extracted to read csv files (default: ./human_robot_interaction_data).')
	parser.add_argument('--dst-dir', type=str, default='./data', metavar='DST',
						help='Path to save the processed trajectories to (default: ./data).')
	parser.add_argument('--downsample-len', type=int, default=250, metavar='NEW_LEN',
						help='Length to downsample trajectories to. If 0, no downsampling is performed (default: 250).')
	parser.add_argument('--crop-zerovel', action="store_true",
						help='Flag to crop zero velocity parts from the beginning and end of trajectories. (default: False).')
	parser.add_argument('--no-augment', action="store_true",
						help='Whether to skip the trajectory augmentation or not. (default: False).')
	args = parser.parse_args()

	preproc(args.src_dir, args.dst_dir, args.crop_zerovel, args.no_augment, args.downsample_len)