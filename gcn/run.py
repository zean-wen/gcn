from __future__ import division
from __future__ import print_function

import time
import json
import tensorflow as tf
from tqdm import tqdm

from gcn.utils import *
from gcn.models import GCN, MLP, GCNModified

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 600, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

flags.DEFINE_string('data_dir', None, 'Data root dir')
flags.DEFINE_string('ids_map_dir', None, 'ids_map dir')
flags.DEFINE_string('tier', 'train', 'train, val, or test')
flags.DEFINE_bool('use_dummy', False, 'use dummy data for test')
flags.DEFINE_string('save_dir', None, 'save dir')
flags.DEFINE_integer('start_index', 0, 'image start index')

ids_map_dir = os.path.join(FLAGS.ids_map_dir, '{}_ids_map.json'.format(FLAGS.tier))
with open(ids_map_dir, 'r') as f:
    n_images = len(json.load(f)['image_ix_to_id'])

for image_index in tqdm(range(n_images)):
    adj, object_name_embeddings, object_visual_features, ocr_bounding_boxes, ocr_token_embeddings, y_train, train_mask = \
        load_data_modified(FLAGS.data_dir, FLAGS.tier, image_index, FLAGS.use_dummy)

    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCNModified

    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'object_name_embeddings': tf.placeholder(tf.float32, shape=(None, object_name_embeddings.shape[1])),
        'object_visual_features': tf.placeholder(tf.float32, shape=(None, object_visual_features.shape[1])),
        'ocr_token_embeddings': tf.placeholder(tf.float32, shape=(None, ocr_token_embeddings.shape[1])),
        'ocr_bounding_boxes': tf.placeholder(tf.float32, shape=(None, ocr_bounding_boxes.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=600, logging=True)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    feed_dict = construct_feed_dict(object_name_embeddings, object_visual_features, ocr_bounding_boxes,
                                    ocr_token_embeddings, support, y_train, train_mask, placeholders)

    # Train model
    for epoch in range(FLAGS.start_index, FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(object_name_embeddings, object_visual_features, ocr_bounding_boxes,
                                        ocr_token_embeddings, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

        # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #     print("Early stopping...")
        #     break

    print("Optimization Finished!")

    feed_dict = construct_feed_dict(object_name_embeddings, object_visual_features, ocr_bounding_boxes,
                                    ocr_token_embeddings, support, y_train, train_mask, placeholders)

    out = sess.run(model.layers[1].hidden_state, feed_dict=feed_dict)

    save_dir = os.path.join(FLAGS.save_dir, '{}_sg_gcn'.format(FLAGS.tier))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.p'.format(image_index)), 'wb') as f:
        pkl.dump(out, f)
