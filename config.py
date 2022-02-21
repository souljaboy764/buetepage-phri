from torch import nn

NUM_JOINTS = 4
JOINTS_DIM = 4
WINDOW_LEN = 40

ROBOT_JOINTS = 7

NUM_ACTIONS = 5

human_vae_config = {
	'num_joints': NUM_JOINTS,
	'joint_dims': JOINTS_DIM,
	'window_size': WINDOW_LEN,
	'hidden_sizes': [250,150],
	'latent_dim': 40,
	'activation': nn.ReLU(),
}

robot_vae_config = {
	'num_joints': ROBOT_JOINTS,
	'joint_dims': 1,
	'window_size': WINDOW_LEN,
	'hidden_sizes': [250,150],
	'latent_dim': 7,
	'activation': nn.ReLU(),
}

tdm_config = {
	'num_joints': NUM_JOINTS,
	'joint_dims': JOINTS_DIM,
	'num_actions': NUM_ACTIONS,
	'encoder_sizes':[256,256,256],
	'latent_dim':40,
	# 'lstm_config': {'input_size':human_vae_config['num_joints']*human_vae_config['joint_dims'] + NUM_ACTIONS, 'hidden_size': 256, 'num_layers': 3},
	'decoder_sizes':[40,40],
	'activation': nn.Tanh(),
	'output_dim':human_vae_config['latent_dim'],
}