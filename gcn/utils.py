import os
import numpy as np
import pickle as pkl
import h5py
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_pkl(tier_data_dir, image_index, use_dummy):
    # node feature loading
    node_feature_dir = os.path.join(tier_data_dir, 'node_features')

    object_name_embedding_dir = os.path.join(node_feature_dir, 'object_name_embeddings')
    image_object_name_embedding_dir = os.path.join(object_name_embedding_dir, '{}.p'.format(image_index))
    with open(image_object_name_embedding_dir, 'rb') as f:
        object_name_embeddings = pkl.load(f)

    object_visual_feature_dir = os.path.join(node_feature_dir, 'object_visual_features')
    image_object_visual_feature_dir = os.path.join(object_visual_feature_dir, '{}.p'.format(image_index))
    with open(image_object_visual_feature_dir, 'rb') as f:
        object_visual_features = pkl.load(f)

    ocr_token_embedding_dir = os.path.join(node_feature_dir, 'ocr_token_embeddings')
    image_ocr_token_embedding_dir = os.path.join(ocr_token_embedding_dir, '{}.p'.format(image_index))
    with open(image_ocr_token_embedding_dir, 'rb') as f:
        ocr_token_embeddings = pkl.load(f)

    ocr_bounding_boxe_dir = os.path.join(node_feature_dir, 'ocr_bounding_boxes')
    image_ocr_bounding_boxe_dir = os.path.join(ocr_bounding_boxe_dir, '{}.p'.format(image_index))
    with open(image_ocr_bounding_boxe_dir, 'rb') as f:
        ocr_bounding_boxes = pkl.load(f)

    # adj matrix loading
    adj_matrix_dir = os.path.join(tier_data_dir, 'adjacent_matrix')
    image_adj_matrix_dir = os.path.join(adj_matrix_dir, '{}.p'.format(image_index))
    with open(image_adj_matrix_dir, 'rb') as f:
        adj = pkl.load(f)
    adj = sp.csr_matrix(adj)

    # target data loading
    target_dir = os.path.join(tier_data_dir, 'targets')
    image_target_dir = os.path.join(target_dir, '{}.p'.format(image_index))
    with open(image_target_dir, 'rb') as f:
        targets = pkl.load(f)

    # use dummy data
    if use_dummy:
        object_name_embeddings = np.random.rand(*object_name_embeddings.shape)
        object_visual_features = np.random.rand(*object_visual_features.shape)
        ocr_bounding_boxes = np.random.rand(*ocr_bounding_boxes.shape)
        ocr_token_embeddings = np.random.rand(*ocr_token_embeddings.shape)
        adj = np.random.rand(*adj.shape)
        adj = sp.csr_matrix(adj)

    idx_train = range(len(targets))

    train_mask = sample_mask(idx_train, targets.shape[0])

    y_train = targets

    return (adj, object_name_embeddings, object_visual_features, ocr_bounding_boxes, ocr_token_embeddings, y_train,
            train_mask)


def load_data_h5(image_index, node_feature_h5, adj_matrix_h5, target_h5, mask_h5):
    object_name_embeddings = node_feature_h5['object_name_embeddings'][image_index]
    object_visual_features = node_feature_h5['object_visual_features'][image_index]
    ocr_bounding_boxes = node_feature_h5['ocr_bounding_boxes'][image_index]
    ocr_token_embeddings = node_feature_h5['ocr_token_embeddings'][image_index]

    adj = adj_matrix_h5['adjacent_matrix'][image_index]
    adj = sp.csr_matrix(adj)

    labels = target_h5['targets'][image_index]

    train_mask = np.array(mask_h5['masks'][image_index], dtype=np.bool)

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    return (adj, object_name_embeddings, object_visual_features, ocr_bounding_boxes, ocr_token_embeddings, y_train,
            train_mask)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(object_name_embeddings, object_visual_features, ocr_bounding_boxes,
                        ocr_token_embeddings, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['object_name_embeddings']: object_name_embeddings})
    feed_dict.update({placeholders['object_visual_features']: object_visual_features})
    feed_dict.update({placeholders['ocr_bounding_boxes']: ocr_bounding_boxes})
    feed_dict.update({placeholders['ocr_token_embeddings']: ocr_token_embeddings})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    # feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['num_features_nonzero']: (1, 600)})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
