import os
import json
import tensorflow as tf
from tqdm import tqdm

from train import train_model

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data', 'Data root dir')
flags.DEFINE_string('save_dir', './genrated_textvqa_gcn_sg', 'dir to save generated scene graph')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 600, 'Number of units in hidden layer, equal to the dim of output scene graph')
flags.DEFINE_string('tiers', 'train_val_test', 'what tier of textvqa sg to use: train, val, or test')
flags.DEFINE_integer('start_index', 0, 'image start index')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_bool('use_dummy', False, 'use dummy data for test')

tiers = FLAGS.tiers.split('_')
for tier in tiers:
	target_dir = os.path.join(FLAGS.data_dir, 'textvqa_{}'.format(tier), 'targets')
	n_images = len(os.listdir(target_dir)) 

	for image_index in tqdm(range(FLAGS.start_index, n_images),
	                        unit='image',
	                        desc='Generating gcn sg for {}'.format(tier)):
		train_model(tier, image_index)
