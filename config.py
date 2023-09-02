class global_config:
	def __init__(self):
		self.NUM_JOINTS = 6
		self.JOINTS_DIM = 3
		self.WINDOW_LEN = 5
		self.ROBOT_JOINTS = 7
		self.NUM_ACTIONS = 5
		self.optimizer = 'AdamW'
		self.lr = 5e-4
		self.EPOCHS = 400
		self.EPOCHS_TO_SAVE = 10

class human_vae_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 100
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [40, 20]
		self.latent_dim = 5
		self.beta = 0.0005
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1

class robot_vae_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 5000
		self.num_joints = config.ROBOT_JOINTS
		self.joint_dims = 1
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [40, 20]
		self.latent_dim = 5
		self.beta = 0.0005
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1

class human_tdm_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 149
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.num_actions = config.NUM_ACTIONS
		self.lstm_hidden = 42
		self.num_lstm_layers = 1
		self.latent_dim = 5
		# 'lstm_config: 'input_size = human_vae_config['num_joints*human_vae_config['joint_dims + NUM_ACTIONS 'hidden_size = 256 'num_layers = 3
		self.decoder_sizes = [5, 5]
		self.activation = 'Tanh'
		self.output_dim = human_vae_config().latent_dim

class hri_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 149
		self.num_joints = config.ROBOT_JOINTS
		self.joint_dims = 1
		self.human_dims = 40
		self.num_actions = config.NUM_ACTIONS
		self.lstm_hidden = 256
		self.linear_hidden = 40
		self.num_lstm_layers = 3
		self.latent_dim = 7
		self.activation = 'Tanh'
		