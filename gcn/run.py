import os
import h5py
import numpy as np
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm

from gcn.train import train_model
from gcn.utils import load_data_pkl, load_data_h5

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('use_h5', True, 'use h5 feature or not')
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

if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)

tiers = FLAGS.tiers.split('_')

for tier in tiers:
    data_root = os.path.join(FLAGS.data_dir, 'textvqa_{}'.format(tier))

    if FLAGS.use_h5:
        node_feature_h5 = h5py.File(os.path.join(data_root, 'node_features.h5'), "r")
        adj_matrix_h5 = h5py.File(os.path.join(data_root, 'adjacent_matrix.h5'), "r")
        target_h5 = h5py.File(os.path.join(data_root, 'targets.h5'), "r")
        mask_h5 = h5py.File(os.path.join(data_root, 'mask.h5'), "r")

        n_images = mask_h5['masks'].shape[0]

        save_h5 = h5py.File(os.path.join(FLAGS.save_dir, '{}_sg.h5'.format(tier)), "w")
        save_h5.create_dataset("gcn_scene_graphs",
                               (n_images,
                                node_feature_h5['object_name_embeddings'].shape[1] +
                                node_feature_h5['ocr_token_embeddings'].shape[1],
                                FLAGS.hidden1),
                               dtype='float32')

        for image_index in tqdm(range(FLAGS.start_index, n_images),
                                unit='image',
                                desc='Generating gcn sg for {}'.format(tier)):
            data = load_data_h5(image_index, node_feature_h5, adj_matrix_h5, target_h5, mask_h5)
            hidden_state = train_model(data)
            save_h5['gcn_scene_graphs'][image_index] = hidden_state
        node_feature_h5.close()
        adj_matrix_h5.close()
        target_h5.close()
        mask_h5.close()
        save_h5.close()

    else:
        target_dir = os.path.join(data_root, 'targets')
        n_images = len(os.listdir(target_dir))
        for image_index in tqdm(range(FLAGS.start_index, n_images),
                                unit='image',
                                desc='Generating gcn sg for {}'.format(tier)):
            data = load_data_pkl(data_root, image_index, FLAGS.use_dummy)
            hidden_state = train_model(data)

            # save 2nd layer hidden state
            save_dir = os.path.join(FLAGS.save_dir, '{}_sg'.format(tier))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir, '{}.p'.format(image_index)), 'wb') as f:
                pkl.dump(hidden_state, f)
