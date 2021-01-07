from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

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
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

flags.DEFINE_string('data_dir', None, 'Data root dir')
flags.DEFINE_string('tier', 'train', 'train, val, or test')
flags.DEFINE_integer('image_index', 0, 'Image Index')


# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

adj, object_name_embeddings, object_visual_features, ocr_bounding_boxes, \
ocr_token_embeddings, y_train, train_mask = load_data_modified(FLAGS.data_dir, FLAGS.tier, FLAGS.image_index)
print(object_name_embeddings)
print(object_visual_features)

# Some preprocessing
# features = preprocess_features(features)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    # model_func = GCN
    model_func = GCNModified
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
# placeholders = {
#     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
#     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
#     'labels_mask': tf.placeholder(tf.int32),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
# }

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


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(object_name_embeddings, object_visual_features, ocr_bounding_boxes,
                                    ocr_token_embeddings, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    # cost_val.append(cost)

    # Print results
    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
    #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
    #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

out = sess.run([model.input_layer.vars['obj_v_proj'], model.input_layer.object_visual_features, model.input_layer.object_embedding], feed_dict=feed_dict)
print(out[0])
print(out[0].shape)
print(out[1])
print(tf.get_variable)
print("Optimization Finished!")

# Testing
# test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
