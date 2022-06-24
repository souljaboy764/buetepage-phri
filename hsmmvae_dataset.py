import pickle, json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import numpy as np
import os
import random
from copy import copy

from scipy.spatial.distance import euclidean


class BuetepageDataset(Dataset):
	def __init__(self, datafile, train=True):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				traj_data = np.array(data['train_data'])
			else:
				traj_data = np.array(data['test_data'])
			self.labels = traj_data[:, -1]
			self.idx = traj_data[:, -2]
			self.traj_data = traj_data[:, :-2]
			self.len = self.traj_data.shape[0]

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index], self.idx[index], self.labels[index]

class BuetepageSequenceDataset(Dataset):
	def __init__(self, datafile, train=True):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				traj_data = np.array(data['train_data'])
			else:
				traj_data = np.array(data['test_data'])
			labels = traj_data[:, -1]
			idx = traj_data[:, -2]
			traj_data = traj_data[:, :-2]
			starts = np.where(idx==0)[0]
			ends = np.array(starts[1:].tolist() + [traj_data.shape[0]])
			traj_seqs = []
			traj_labels = []
			for i in range(len(starts)):
				traj_seqs.append(traj_data[starts[i]:ends[i]])
				traj_labels.append(labels[starts[i]:ends[i]])

			self.len = len(starts)
			self.traj_data = traj_seqs
			self.labels = traj_labels

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index], np.arange(self.traj_data[index].shape[0]), self.labels[index]

# class SkeletonSequenceDataset(Dataset):
# 	def __init__(self, root_dir='/home/elenoide/ssd_data/vignesh/amass/dataset/CMU/', subject = 143, bm_path = '/home/elenoide/ssd_data/vignesh/amass/smplh/neutral/model.npz', transform=None, upperbody_only=False):
# 		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		self.transform = transform
# 		self.root_dir = root_dir
# 		fnames = os.listdir(os.path.join(root_dir, str(subject)))
# 		self.data = []
# 		self.time_steps = []
# 		lens = []
# 		self.clipped_data = []
# 		for i in range(len(fnames)):
# 			f = fnames[i]
# 			data = np.load(os.path.join(root_dir,str(subject),f))
# 			bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=data['poses'].shape[0]).to(device)
# 			body_v = bm(pose_body=torch.tensor(data['poses'][:, 3:66]).to(torch.float32).to(device), betas=torch.tensor(data['betas'][:10][np.newaxis]).to(device).to(torch.float32)).Jtr[:,:-30].cpu().detach().numpy().reshape(data['poses'].shape[0],-1)
# 			# self.betas = torch.tensor(data['betas'][:10][np.newaxis]).to(torch.float32)
# 			# body_v = data['poses'][:, :66]
# 			lens.append(body_v.shape[0])
# 			for j in range(60,len(body_v),60):
# 				self.clipped_data.append(body_v[j-60:j])
# 			self.data.append(body_v.astype(np.float32))
# 			self.time_steps.append(np.linspace(0,1,data['poses'].shape[0]).astype(np.float32))
# 			# else:
# 			# 	self.data = np.vstack([self.data, data['poses'][:,:66]])
# 			# 	self.time_steps = np.vstack([self.time_steps, np.linspace(0,1,data['poses'].shape[0])])
# 				# self.data = self.data + data['poses'][:,:66].tolist()
# 		# for i in range(1,len(self.clipped_data)):
# 		# 	_, path = fastdtw(self.clipped_data[0], self.clipped_data[i], dist=euclidean)
# 		# 	path = np.array(path)
# 		# 	self.clipped_data[i] = self.clipped_data[i][path[:,1]]

# 		# min_len = np.min(lens)
# 		# for i in range(len(fnames)):

# 		# self.clipped_data = self.clipped_data
# 		# print(self.clipped_data.shape)


# 		# self.data = self.data.astype(np.float32)
# 		# self.time_steps = self.time_steps.astype(np.float32)
# 		# shape = self.data.shape 
# 		# self.time_steps = np.zeros(shape[:-1],dtype=int)
# 		# for i in range(shape[0]):
# 		# 	self.labels[i,:,:] = i
# 		# self.labels = self.labels.reshape((-1,shape[-2]))
# 		# for i in range(shape[2]):
# 		# 	self.time_steps[:,:,i] = i
# 		# self.time_steps = self.time_steps.reshape((-1,shape[-2]))
# 		# self.frames_per_sequence = shape[2] 
# 		# if upperbody_only:
# 		# 	# self.data = self.data[:,:,[0, 3, 5, 6, 8, 9, 10, 12, 13, 14]]
# 		# 	self.data = self.data.reshape(-1, shape[-2], shape[-1]//3, 3)
# 		# 	self.data = self.data[:,:,upperbody_idx].reshape(-1, shape[-2], len(upperbody_idx)*3)

# 	def __len__(self):
# 		return len(self.clipped_data)

# 	def __getitem__(self, index):
# 		frame = self.clipped_data[index]
		
# 		if self.transform:
# 			frame = self.transform(frame)
		
# 		# return frame, len(self.sequences[index])
# 		return frame

# class SkeletonSequenceDataset(Dataset):
# 	def __init__(self, root_file='/home/elenoide/ssd_data/vignesh/cmuseq143_nonaligned.npz', transform=None, upperbody_only=False):
# 		self.transform = transform
# 		self.data = np.load(root_file,allow_pickle=True)['arr_0']
# 		self.time_steps = []
# 		for i in range(len(self.data)):
# 			self.time_steps.append(np.linspace(0,1,self.data[i].shape[0]).astype(np.float32))
# 		self.time_steps=np.array(self.time_steps)
			
# 	def __len__(self):
# 		return len(self.data)

# 	def __getitem__(self, index):
# 		frame = self.data[index]
		
# 		if self.transform:
# 			frame = self.transform(frame)
		
# 		return frame

# class SkeletonSequenceAlignedDataset(Dataset):
# 	def __init__(self, root_file='/home/elenoide/ssd_data/vignesh/cmuseq143_aligned.npz', transform=None, upperbody_only=False):
# 		self.transform = transform
# 		self.data = np.load(root_file,allow_pickle=True)['arr_0']
# 		self.time_steps = []
# 		for i in range(len(self.data)):
# 			self.time_steps.append(np.linspace(0,1,self.data[i].shape[0]).astype(np.float32))
			
# 			# if upperbody_only:
# 			# 	# self.data = self.data[:,:,[0, 3, 5, 6, 8, 9, 10, 12, 13, 14]]
# 			# 	self.data = self.data.reshape(-1, shape[-2], shape[-1]//3, 3)
# 			# 	self.data = self.data[:,:,upperbody_idx].reshape(-1, shape[-2], len(upperbody_idx)*3)

# 	def __len__(self):
# 		return len(self.data)

# 	def __getitem__(self, index):
# 		frame = self.data[index]
		
# 		if self.transform:
# 			frame = self.transform(frame)
		
# 		# return frame, len(self.sequences[index])
# 		return frame


# if __name__=='__main__':
# 	dataset = SkeletonSequenceDataset()
# 	# print(len(dataset.data))
# 	# for i in range(len(dataset.data)):
# 	# 	if len(dataset.data[i])!=60:
# 	# 		print(i, len(dataset.data[i]))
# 	np.savez_compressed('/home/elenoide/ssd_data/vignesh/cmuseq143_nonaligned.npz',dataset.clipped_data)
