class global_config:
	def __init__(self):
		self.NUM_JOINTS = 4
		self.JOINTS_DIM = 3
		self.WINDOW_LEN = 40
		self.ROBOT_JOINTS = 7
		self.NUM_ACTIONS = 5
		self.optimizer = 'Adam'
		self.lr = 0.00001
		self.EPOCHS = 3000
		self.EPOCHS_TO_SAVE = 100

class human_vae_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 32
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [250, 150]
		self.latent_dim = 40
		self.beta = 0.5
		self.activation = 'ReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1

class human_vrnn_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 150
		self.num_joints = 3
		self.joint_dims = config.JOINTS_DIM
		self.hidden_sizes = [50]
		self.latent_dim = 3
		self.beta = 0.5
		self.activation = 'ReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1

class human_simplevrnn_config:
	def __init__(self):
		config = global_config()
		self.reuse_encoder = True
		self.batch_size = 150
		self.num_joints = 3
		self.joint_dims = config.JOINTS_DIM
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [250, 150]
		self.latent_dim = 40
		self.beta = 0.5
		self.activation = 'ReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1

class robot_vae_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 5000
		self.num_joints = config.ROBOT_JOINTS
		self.joint_dims = 1
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [250, 150]
		self.latent_dim = 7
		self.activation = 'ReLU'

class human_tdm_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 149
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.num_actions = config.NUM_ACTIONS
		self.lstm_hidden = 256
		self.num_lstm_layers = 3
		self.latent_dim = 40
		# 'lstm_config: 'input_size = human_vae_config['num_joints*human_vae_config['joint_dims + NUM_ACTIONS 'hidden_size = 256 'num_layers = 3
		self.decoder_sizes = [40, 40]
		self.activation = 'Tanh'
		self.output_dim = human_vae_config().latent_dim
