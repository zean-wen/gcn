import os
import h5py
import json

from .config import Config, get_args
from .utils import delete_zero_padding
from .node_feature import NodeFeature
from .adj_matrix import AdjMatrix
from .target import Target
from .mask import Mask


class DataSet:
    def __init__(self,
                 tier,
                 save_dir,
                 save_h5,
                 data_root,
                 word_emb_config):
        self.tier = tier

        # create data save dir
        save_dir = os.path.join(save_dir, 'textvqa_{}'.format(tier))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # read ids map
        print("Loading data...")
        ids_map_dir = os.path.join(data_root,
                                   'ids_map',
                                   '{}_ids_map.json'.format(tier))
        with open(ids_map_dir, 'r') as f:
            image_ix_to_id = json.load(f)['image_ix_to_id']
        n_images = len(image_ix_to_id)

        # read node json file
        node_dir = os.path.join(data_root,
                                'nodes',
                                '{}_nodes.json'.format(tier))
        with open(node_dir, 'r') as f:
            nodes = json.load(f)
        n_nodes = len(nodes)
        image_n_nodes = {}  # recode number of node for each image
        for image_index in range(n_images):
            image_id = image_ix_to_id[str(image_index)]
            image_n_nodes[image_index] = len(nodes[image_id])

        # read visual feature h5 file
        visual_feature_dir = os.path.join(data_root,
                                          'object_visual_feature',
                                          '{}_objects.h5'.format(tier))
        visual_feature_h5 = h5py.File(visual_feature_dir, 'r')
        n_objects = len(visual_feature_h5['features'])
        image_n_objects = {}
        for image_index in range(n_images):
            image_n_objects[image_index] = len(delete_zero_padding(visual_feature_h5['features'][image_index]))

        # read ocr
        if tier == 'val':
            ocr_dir = os.path.join(data_root,
                                   'ocr',
                                   '{}_ocr.json'.format('train'))
        else:
            ocr_dir = os.path.join(data_root,
                                   'ocr',
                                   '{}_ocr.json'.format(tier))
        with open(ocr_dir, 'r') as f:
            ocr = json.load(f)
        # n_ocr = len(ocr)
        max_n_ocr = 0
        image_n_ocr = {}
        for image_index in range(n_images):
            image_id = image_ix_to_id[str(image_index)]
            image_n_ocr[image_index] = len(ocr[image_id])
            if max_n_ocr < len(ocr[image_id]):
                max_n_ocr = len(ocr[image_id])

        # read adjacent matrix
        adj_matrix_dir = os.path.join(data_root,
                                      'adjacent_matrix',
                                      '{}_edge_rdiou.json'.format(tier))
        with open(adj_matrix_dir, 'r') as f:
            adj_matrix = json.load(f)
        image_adj_dim = {}
        for image_index in range(n_images):
            image_id = image_ix_to_id[str(image_index)]
            image_adj_dim[image_index] = len(adj_matrix[image_id])

        # check input data correctness
        assert n_images == n_nodes
        assert n_images == n_objects
        # assert n_images == n_ocr
        for image_index in range(n_images):
            assert image_n_nodes[image_index] == image_adj_dim[image_index]
            assert (image_n_objects[image_index] + image_n_ocr[image_index]) == image_n_nodes[image_index]

        self.node_feature = NodeFeature(save_dir,
                                        save_h5,
                                        image_ix_to_id,
                                        n_images,
                                        nodes,
                                        image_n_objects,
                                        ocr,
                                        max_n_ocr,
                                        visual_feature_dir,
                                        word_emb_config)
        self.adjacent_matrix = AdjMatrix(save_dir,
                                         save_h5,
                                         image_ix_to_id,
                                         n_images,
                                         adj_matrix,
                                         max_n_ocr,
                                         image_n_objects,
                                         image_n_ocr)
        self.target = Target(save_dir,
                             save_h5,
                             n_images,
                             image_n_objects,
                             image_n_ocr,
                             max_n_ocr,
                             ocr)

        self.mask = Mask(save_dir,
                         save_h5,
                         n_images,
                         max_n_ocr,
                         image_n_objects,
                         image_n_ocr)

    def generate(self):
        print('#### Generating graph data for {} images ####'.format(self.tier))
        self.node_feature.generate()
        self.adjacent_matrix.generate()
        self.target.generate()
        self.mask.generate()


def main():
    args = get_args()
    config = Config(args)

    tiers = config.tiers.split('_')
    for tier in tiers:
        data_set = DataSet(tier,
                           config.save_dir,
                           config.save_h5,
                           config.data_root,
                           config.word_emb_config)
        data_set.generate()


if __name__ == '__main__':
    main()
