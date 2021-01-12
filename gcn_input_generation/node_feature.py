import os
from tqdm import tqdm
import h5py
import pickle
import numpy as np

from .utils import WordEmbedding, delete_zero_padding


class NodeFeature:
    def __init__(self,
                 save_dir,
                 image_ix_to_id,
                 n_images,
                 nodes,
                 image_n_objects,
                 ocr,
                 visual_feature_h5,
                 word_emb_config):

        self.image_ix_to_id = image_ix_to_id
        self.nodes = nodes
        self.image_n_objects = image_n_objects
        self.visual_feature_h5 = visual_feature_h5
        self.ocr = ocr
        self.n_images = n_images
        self.word_emb_config = word_emb_config

        node_feature_dir = os.path.join(save_dir, 'node_features')
        if not os.path.exists(node_feature_dir):
            os.mkdir(node_feature_dir)
        self.dir = {'object_name_embeddings': os.path.join(node_feature_dir, 'object_name_embeddings'),
                    'object_visual_features': os.path.join(node_feature_dir, 'object_visual_features'),
                    'ocr_token_embeddings': os.path.join(node_feature_dir, 'ocr_token_embeddings'),
                    'ocr_bounding_boxes': os.path.join(node_feature_dir, 'ocr_bounding_boxes')}
        for path in self.dir.values():
            if not os.path.exists(path):
                os.mkdir(path)

    def generate(self):
        self.object_name_embedding_generation()
        self.object_visual_feature_generation()
        self.ocr_feature_generation()

    def object_name_embedding_generation(self):
        word_embed = WordEmbedding('glove', self.word_emb_config)
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Object name embedding generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_nodes = self.nodes[image_id]
            n_objects = self.image_n_objects[image_index]
            image_object_name_embeddings = np.zeros((n_objects, 300), dtype='float32')
            for object_index in range(n_objects):
                image_object_name_embeddings[object_index] = word_embed(image_nodes[object_index])
            with open(os.path.join(self.dir['object_name_embeddings'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_object_name_embeddings, f)

    def object_visual_feature_generation(self):
        object_visual_features = h5py.File(self.visual_feature_h5, 'r')
        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Object visual feature generation'):
            image_object_visual_features = delete_zero_padding(object_visual_features['features'][image_index])
            with open(os.path.join(self.dir['object_visual_features'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_object_visual_features, f)
        object_visual_features.close()

    def ocr_feature_generation(self):
        word_embed = WordEmbedding('fasttext', self.word_emb_config)

        for image_index in tqdm(range(self.n_images),
                                unit='image',
                                desc='Ocr feature generation'):
            image_id = self.image_ix_to_id[str(image_index)]
            image_ocr = self.ocr[image_id]
            n_ocr = len(image_ocr)
            image_ocr_token_embeddings = np.zeros((n_ocr, 300), dtype='float32')
            image_ocr_bounding_boxes = np.zeros((n_ocr, 8), dtype='float32')
            for ocr_index, (ocr_token, bbox) in enumerate(image_ocr.items()):
                image_ocr_token_embeddings[ocr_index] = word_embed(ocr_token)
                image_ocr_bounding_boxes[ocr_index] = np.array(bbox).flatten()
            with open(os.path.join(self.dir['ocr_token_embeddings'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_ocr_token_embeddings, f)
            with open(os.path.join(self.dir['ocr_bounding_boxes'], '{}.p'.format(image_index)), 'wb') as f:
                pickle.dump(image_ocr_bounding_boxes, f)
