from __future__ import division
from __future__ import print_function

import tensorflow as tf

from gcn.utils import *
from gcn.models import *


def train_model(data):
    adj, object_name_embeddings, object_visual_features, ocr_bounding_boxes, ocr_token_embeddings, y_train, \
    train_mask = data

    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCNModified

    placeholders = {'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                    'object_name_embeddings': tf.placeholder(tf.float32, shape=(None, object_name_embeddings.shape[1])),
                    'object_visual_features': tf.placeholder(tf.float32, shape=(None, object_visual_features.shape[1])),
                    'ocr_token_embeddings': tf.placeholder(tf.float32, shape=(None, ocr_token_embeddings.shape[1])),
                    'ocr_bounding_boxes': tf.placeholder(tf.float32, shape=(None, ocr_bounding_boxes.shape[1])),
                    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
                    'labels_mask': tf.placeholder(tf.int32),
                    'dropout': tf.placeholder_with_default(0., shape=()),
                    'num_features_nonzero': tf.placeholder(tf.int32)}

    # Create model
    model = model_func(placeholders, input_dim=600, logging=True)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model
    for epoch in range(FLAGS.epochs):
        # Construct feed dictionary
        feed_dict = construct_feed_dict(object_name_embeddings, object_visual_features, ocr_bounding_boxes,
                                        ocr_token_embeddings, support, y_train, train_mask, placeholders)
        # add drop out 
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # get 2nd layer hidden state
    feed_dict = construct_feed_dict(object_name_embeddings, object_visual_features, ocr_bounding_boxes,
                                    ocr_token_embeddings, support, y_train, train_mask, placeholders)

    out = sess.run(model.layers[1].hidden_state, feed_dict=feed_dict)

    return out
