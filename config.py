class global_config:
	def __init__(self):
		self.NUM_JOINTS = 4
		self.JOINTS_DIM = 3
		self.WINDOW_LEN = 40
		self.ROBOT_JOINTS = 7
		self.NUM_ACTIONS = 5
		self.optimizer = 'AdamW'
		self.lr = 0.0001
		self.EPOCHS = 2000
		self.EPOCHS_TO_SAVE = 100

class human_vae_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 5000
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [250, 150]
		self.latent_dim = 40
		self.beta = 0.001
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


class tdm_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 500
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.num_actions = config.NUM_ACTIONS
		self.encoder_sizes = [256, 256, 256]
		self.latent_dim = 40
		# 'lstm_config: 'input_size = human_vae_config['num_joints*human_vae_config['joint_dims + NUM_ACTIONS 'hidden_size = 256 'num_layers = 3
		self.decoder_sizes = [40, 40]
		self.activation = 'Tanh'
		self.output_dim = human_vae_config().latent_dim
