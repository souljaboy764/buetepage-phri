import torch
from torch.utils.data import Dataset
import numpy as np

# class SkeletonDataset(Dataset):
# 	def __init__(self, datafile, train=True):
# 		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		with np.load(datafile, allow_pickle=True) as data:
# 			if train:
# 				traj_data = np.array(data['train_data'])
# 				self.actidx = np.array([[0,24],[24,54],[54,110],[110,149]])
# 			else:
# 				traj_data = np.array(data['test_data'])
# 				self.actidx = np.array([[0,7],[7,15],[15,29],[29,39]])
# 			self.labels = traj_data[:, -1]
# 			self.idx = traj_data[:, -2]
# 			self.traj_data = traj_data[:, :-2]
# 			self.len = self.traj_data.shape[0]
# 			starts = np.where(self.idx==0)[0]
# 			ends = np.array(starts[1:].tolist() + [traj_data.shape[0]])
# 			self.traj_lens = np.zeros_like(self.idx)
# 			for i in range(len(starts)):
# 				self.traj_lens[starts[i]:ends[i]] = ends[i] - starts[i]


# 	def __len__(self):
# 		return self.len

# 	def __getitem__(self, index):
# 		return self.traj_data[index], self.idx[index], self.labels[index], self.traj_lens[index]

class SequenceDataset(Dataset):
	def __init__(self, datafile, train=True):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				self.traj_data = data['train_data']
				self.labels = data['train_labels']
				self.actidx = np.array([[0,24],[24,54],[54,110],[110,149]])
				# self.actidx = np.array([[0,8],[8,16],[16,24],[24,32]]) # Human-robot trajs
			else:
				self.traj_data = data['test_data']
				self.labels = data['test_labels']
				self.actidx = np.array([[0,7],[7,15],[15,29],[29,39]])
				# self.actidx = np.array([[0,2],[2,4],[4,6],[6,9]]) # Human-robot trajs
			
			self.len = len(self.traj_data)
			self.labels = np.zeros(self.len)
			for idx in range(len(self.actidx)):
				self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)

class SequenceWindowDataset(Dataset):
	def __init__(self, datafile, train=True, window_length=40):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				traj_data = data['train_data']
				labels = data['train_labels']
				self.actidx = np.array([[0,24],[24,54],[54,110],[110,149]])
				# self.actidx = np.array([[0,8],[8,16],[16,24],[24,32]]) # Human-robot trajs
			else:
				traj_data = data['test_data']
				labels = data['test_labels']
				self.actidx = np.array([[0,7],[7,15],[15,29],[29,39]])
				# self.actidx = np.array([[0,2],[2,4],[4,6],[6,9]])

			self.traj_data = []
			self.labels = []
			for i in range(len(traj_data)):
				trajs_concat = []
				traj_shape = traj_data[i].shape
				dim = traj_shape[-1]
				for traj in [traj_data[i][:,:dim//2], traj_data[i][:,dim//2:]]:
					idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - window_length)])
					trajs_concat.append(traj[idx].reshape((traj_shape[0] + 1 - window_length, window_length*dim//2)))
				trajs_concat = np.concatenate(trajs_concat,axis=-1)
				self.traj_data.append(trajs_concat)
				self.labels.append(labels[i][:trajs_concat.shape[0]])
			
			self.len = len(self.traj_data)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)